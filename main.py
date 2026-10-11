"""
Modul Main
Titik masuk dari aplikasi Iron Man Controller.
Membuka kamera, membaca frame, dan menggabungkan semua modul.
"""
import cv2
import time
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from controller import SystemController

def main():
    cap = cv2.VideoCapture(0)
    
    # =========================================================================
    # OPTICAL ENHANCEMENT: True Widescreen 16:9 HD & Zero-Lag Buffer
    # Beralih ke 1280x720 membuka sensor widescreen native webcam (Zoom Out alami),
    # memberikan bidang pandang (FOV) yang jauh lebih luas bagi pergerakan tangan.
    # =========================================================================
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Memangkas antrean frame ke 0 ms latency
    cap.set(cv2.CAP_PROP_FPS, 60)

    tracker = HandTracker(detection_con=0.8, track_con=0.8)
    recognizer = GestureRecognizer()
    controller = SystemController()

    window_name = "Iron Man Vision Controller"
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    # Default ukuran jendela luas (Widescreen 960x540) yang proporsional dan terasa lega
    cv2.resizeWindow(window_name, 960, 540)
    show_window = True

    print("==================================================")
    print("IRON MAN COMPUTER VISION CONTROLLER AKTIF!")
    print("==================================================")
    print("Mekanika Tuas Gas Pergelangan (Wrist Tilt Throttle):")
    print("  1. Acungkan Telunjuk untuk mengunci kemiringan netral.")
    print("  2. LENGAN DIAM! Cukup Tundukkan/Tekuk pergelangan Anda:")
    print("     - Tekuk telapak ke BAWAH (layaknya ngegas) = Scroll BAWAH")
    print("     - Tengadahkan telapak ke ATAS = Scroll ATAS")
    print("  3. Buka seluruh tangan (Rentangkan Jari) untuk mengerem.")
    print("==================================================")
    print("Fitur UI & Mode Latar Belakang:")
    print("  - Kualitas Video HD 720p Widescreen (Zoom Out & Lapang).")
    print("  - Jendela Kamera Selalu di Atas (Always on Top).")
    print("  - Klik silang 'X' untuk masuk Mode Latar Belakang (Headless).")
    print("  - Tekan 'q' pada jendela ATAU Ctrl+C di terminal untuk keluar.")
    print("==================================================")

    p_time = 0

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            img = tracker.find_hands(img)
            lm_list = tracker.get_position(img)

            status_text = "NO HAND"
            current_tilt_val = None

            if len(lm_list) != 0:
                is_active, current_tilt = recognizer.get_joystick_state(lm_list)
                current_tilt_val = current_tilt
                status_text = controller.process_dynamic_motion(is_active, current_tilt)
                
                if is_active:
                    # Ambil Ujung Telunjuk dan Pergelangan Tangan
                    index_tip = lm_list[8]
                    wrist = lm_list[0]
                    
                    cx, cy = index_tip[1], index_tip[2]
                    wx, wy = wrist[1], wrist[2]
                    
                    # =========================================================
                    # VISUALISASI SCI-FI HOLOGRAPHIC (JARVIS REPULSOR BEAM)
                    # =========================================================
                    # Garis Laser Tuas Gas (Neon Cyan Glow)
                    cv2.line(img, (wx, wy), (cx, cy), (255, 230, 0), 3, cv2.LINE_AA)
                    # Engsel Pergelangan Tangan (Gold Arc Anchor)
                    cv2.circle(img, (wx, wy), 10, (0, 165, 255), cv2.FILLED, cv2.LINE_AA)
                    cv2.circle(img, (wx, wy), 14, (255, 255, 255), 1, cv2.LINE_AA)
                    # Reticle Target Hologram di Ujung Telunjuk (Arc Reactor Dual-Ring)
                    cv2.circle(img, (cx, cy), 8, (0, 255, 255), cv2.FILLED, cv2.LINE_AA)
                    cv2.circle(img, (cx, cy), 16, (255, 220, 0), 2, cv2.LINE_AA)
                    cv2.circle(img, (cx, cy), 22, (0, 215, 255), 1, cv2.LINE_AA)
                    
            else:
                recognizer.reset_smoothing()
                status_text = controller.process_dynamic_motion(False, None)

            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time

            if show_window:
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        show_window = False
                        cv2.destroyWindow(window_name)
                        print("\n[INFO] Jendela ditutup. Sistem berjalan mulus di LATAR BELAKANG (Headless Mode).")
                        print("[INFO] Tekan 'Ctrl + C' di terminal ini untuk mematikan program sepenuhnya.")
                except cv2.error:
                    show_window = False
                    
            if show_window:
                # =============================================================
                # GLASSMORPHIC HUD TOP BAR (Translucent Futuristic Overlay)
                # =============================================================
                hud_h = 88
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, hud_h), (12, 16, 24), cv2.FILLED)
                cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
                # Garis aksen bawah HUD (Neon Cyan Accent)
                cv2.line(img, (0, hud_h), (w, hud_h), (255, 200, 0), 2, cv2.LINE_AA)

                # Warna status badge dinamis
                status_color = (0, 255, 120) if ("ENGAGED" in status_text or "SCROLL" in status_text or "SWIPED" in status_text) else (0, 200, 255)
                if status_text == "NO HAND":
                    status_color = (60, 60, 230)

                # Judul Sistem & Telemetri Kiri
                cv2.putText(img, "JARVIS VISION // MARK VII CONTROLLER", (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 215, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"STATUS: {status_text}", (20, 66), cv2.FONT_HERSHEY_DUPLEX, 0.85, status_color, 2, cv2.LINE_AA)
                
                # Telemetri Kanan (FPS & Throttle Status)
                cv2.putText(img, f"FPS: {int(fps)}", (w - 150, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
                
                if controller.is_engaged and controller.anchor_tilt is not None and current_tilt_val is not None:
                    offset = current_tilt_val - controller.anchor_tilt
                    tilt_info = f"Offset: {offset:+.1f} (DZ: {controller.deadzone})"
                elif current_tilt_val is not None:
                    tilt_info = f"Tilt: {current_tilt_val:.1f}"
                else:
                    tilt_info = "STANDBY"
                cv2.putText(img, f"THROTTLE: {tilt_info}", (w - 410, 66), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                
                cv2.imshow(window_name, img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                pass

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan paksa oleh pengguna melalui Terminal (Ctrl+C).")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Program dimatikan secara aman.")

if __name__ == "__main__":
    main()
