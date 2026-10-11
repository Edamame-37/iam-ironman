"""
Modul SystemController
Mengeksekusi pergerakan scroll dengan mekanika Kemiringan Tuas Gas Pergelangan (Wrist Tilt).
"""
import pyautogui
import pygetwindow as gw

class SystemController:
    # Deadzone diatur pada 4 derajat rotasi fisik.
    # Karena input kemiringan telah menggunakan sudut 3D derajat murni dan disaring oleh
    # 1 Euro Filter, nilai 4 derajat memberikan respon cepat namun kebal dari tremor otot.
    def __init__(self, sensitivity=0.8, deadzone=4):
        self.sensitivity = sensitivity 
        self.deadzone = deadzone       
        
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        
        self.is_engaged = False
        self.anchor_tilt = None
        self.has_paged_in_current_swipe = False 
        
        self.smoothed_speed = 0.0
        self.speed_ema_alpha = 0.4 # Akselerasi scroll yang responsif

    def is_presentation_active(self):
        try:
            window = gw.getActiveWindow()
            if window and window.title:
                title = window.title.lower()
                keywords = ["powerpoint", "slide", "presentasi", "presentation", "canva"]
                return any(kw in title for kw in keywords)
        except Exception:
            pass
        return False

    def process_dynamic_motion(self, is_active, current_tilt):
        if current_tilt is None:
            self._reset_state()
            return "IDLE"

        is_presentation = self.is_presentation_active()

        if is_active:
            if not self.is_engaged:
                self.is_engaged = True
                self.anchor_tilt = current_tilt 
                self.has_paged_in_current_swipe = False
                self.smoothed_speed = 0.0
                return "ENGAGED (TILT LOCKED)"
            else:
                offset_tilt = current_tilt - self.anchor_tilt
                
                if is_presentation:
                    swipe_threshold = 25 
                    if not self.has_paged_in_current_swipe:
                        if offset_tilt < -swipe_threshold:
                            pyautogui.press('pagedown')
                            self.has_paged_in_current_swipe = True
                            return "SWIPED: NEXT SLIDE"
                        elif offset_tilt > swipe_threshold:
                            pyautogui.press('pageup')
                            self.has_paged_in_current_swipe = True
                            return "SWIPED: PREV SLIDE"
                else:
                    if abs(offset_tilt) > self.deadzone:
                        raw_speed = (abs(offset_tilt) - self.deadzone) * self.sensitivity
                        self.smoothed_speed = (self.speed_ema_alpha * raw_speed) + ((1 - self.speed_ema_alpha) * self.smoothed_speed)
                        
                        final_speed = int(self.smoothed_speed)
                        if final_speed < 1:
                            final_speed = 1
                            
                        if offset_tilt < 0:
                            pyautogui.scroll(final_speed) 
                            return f"AUTO-SCROLL UP ({final_speed})"
                        else:
                            pyautogui.scroll(-final_speed) 
                            return f"AUTO-SCROLL DOWN ({final_speed})"
                    else:
                        self.smoothed_speed = (1 - self.speed_ema_alpha) * self.smoothed_speed
                
                return "HOLDING TILT ANCHOR"
        else:
            self._reset_state()
            return "IDLE"

    def _reset_state(self):
        self.is_engaged = False
        self.anchor_tilt = None
        self.has_paged_in_current_swipe = False
        self.smoothed_speed = 0.0
