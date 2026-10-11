"""
Modul HandTracker (Versi Modern - Tasks API)
Bertugas untuk mendeteksi tangan dan mendapatkan posisi landmark menggunakan MediaPipe Tasks API.
"""
import cv2
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.7):
        """
        Inisialisasi HandTracker dengan parameter MediaPipe Tasks API.
        """
        self.model_path = 'hand_landmarker.task'
        # Download model hand_landmarker.task secara otomatis jika belum ada di komputer
        if not os.path.exists(self.model_path):
            print("\nMendownload model AI deteksi tangan (hanya perlu dilakukan sekali)...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, self.model_path)
            print("Download selesai!\n")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        
        # Menggunakan mode IMAGE karena diproses sinkron per frame (cocok untuk loop sederhana)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=float(detection_con),
            min_hand_presence_confidence=float(track_con),
            min_tracking_confidence=float(track_con),
            running_mode=vision.RunningMode.IMAGE
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        
        # Definisi manual koneksi tulang jari (pengganti modul solutions yang telah dihapus Google)
        self.HAND_CONNECTIONS = [
            (0,1), (1,2), (2,3), (3,4),       # Jempol
            (0,5), (5,6), (6,7), (7,8),       # Telunjuk
            (5,9), (9,10), (10,11), (11,12),  # Tengah
            (9,13), (13,14), (14,15), (15,16),# Manis
            (13,17), (17,18), (18,19), (19,20),# Kelingking
            (0,17)                            # Telapak bawah
        ]

    def find_hands(self, img, draw=True):
        """
        Mendeteksi tangan dalam frame gambar.
        Mengembalikan frame asli yang sudah digambar landmark-nya jika draw=True.
        """
        # Konversi BGR ke RGB untuk MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Konversi dari NumPy Array ke Format Objek Gambar MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Deteksi tangan menggunakan Tasks API terbaru
        self.results = self.detector.detect(mp_image)

        # Gambar manual di atas OpenCV (Pengganti mp_draw.draw_landmarks)
        if self.results and self.results.hand_landmarks and draw:
            for hand_lms in self.results.hand_landmarks:
                h, w, c = img.shape
                pts = {}
                
                # Simpan titik koordinat & Gambar titik
                for id, lm in enumerate(hand_lms):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pts[id] = (cx, cy)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                # Gambar garis penghubung sendi tangan
                for p1, p2 in self.HAND_CONNECTIONS:
                    if p1 in pts and p2 in pts:
                        cv2.line(img, pts[p1], pts[p2], (0, 255, 0), 2)
        
        return img

    def get_position(self, img, hand_no=0):
        """
        Mendapatkan posisi sendi/landmark tangan yang terdeteksi dalam koordinat 2D dan 3D.
        Mengembalikan list berisi:
        [id, cx, cy, cz, norm_x, norm_y, norm_z]
        - id: Index sendi MediaPipe (0-20)
        - cx, cy: Koordinat piksel layar (0 hingga w, 0 hingga h)
        - cz: Estimasi kedalaman optik dalam skala piksel
        - norm_x, norm_y, norm_z: Koordinat 3D murni ternormalisasi dari MediaPipe (invarian terhadap resolusi)
        """
        lm_list = []
        if self.results and self.results.hand_landmarks:
            # Memastikan index hand_no tidak lebih dari jumlah tangan terdeteksi
            if hand_no < len(self.results.hand_landmarks):
                my_hand = self.results.hand_landmarks[hand_no]
                h, w, c = img.shape
                for id, lm in enumerate(my_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cz = int(lm.z * w)
                    # Menyimpan koordinat piksel 2D, kedalaman cz, dan koordinat 3D asli ternormalisasi
                    lm_list.append([id, cx, cy, cz, lm.x, lm.y, lm.z])
        return lm_list

