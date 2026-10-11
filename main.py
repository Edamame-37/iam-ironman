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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = HandTracker(detection_con=0.8, track_con=0.8)
    recognizer = GestureRecognizer()
    controller = SystemController()

    window_name = "Iron Man Vision Controller"
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
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
            img = tracker.find_hands(img)
            lm_list = tracker.get_position(img)

            status_text = "NO HAND"
            conf_val = recognizer.last_confidence
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
                    
                    # Visualisasi Tuas Gas (Garis yang menghubungkan engsel pergelangan ke telunjuk)
                    cv2.line(img, (wx, wy), (cx, cy), (0, 255, 255), 4) # Garis Kuning Penghubung
                    cv2.circle(img, (wx, wy), 10, (255, 0, 0), cv2.FILLED) # Titik Biru di Engsel
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED) # Titik Hijau di Ujung
                    
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
                # Warna status dinamis
                status_color = (0, 255, 0) if "ENGAGED" in status_text or "SCROLL" in status_text or "SWIPED" in status_text else (0, 200, 255)
                if status_text == "NO HAND":
                    status_color = (0, 0, 255)

                cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 0, 0), 2)
                cv2.putText(img, f'Status: {status_text}', (10, 65), cv2.FONT_HERSHEY_PLAIN, 1.8, status_color, 2)
                
                # Informasi Probabilitas & Sudut Derajat
                conf_pct = int(recognizer.last_confidence * 100)
                tilt_info = f"{current_tilt_val:.1f} deg" if current_tilt_val is not None else "-"
                cv2.putText(img, f'Confidence: {conf_pct}% | Pitch: {tilt_info}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 200, 200), 2)
                
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
