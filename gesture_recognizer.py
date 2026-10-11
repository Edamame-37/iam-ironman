"""
Modul GestureRecognizer (Versi 3D Phalanx Ratio & Linear Normalized Throttle)
Menganalisis pergerakan kemiringan telapak tangan (Wrist Tilt Throttle) menggunakan:
1. 3D Phalanx Extension Ratio (100% Invarian terhadap perspektif kamera dan sudut kemiringan tangan)
2. Linear Normalized Pitch Throttle (Bebas dari singularitas matematis arcsin, respon 1:1 mulus)
3. Bi-directional Debouncing (Mencegah false positive saat mengetik dan mencegah false drop saat scrolling)
4. 1 Euro Filter Adaptif (Membasmi getaran saat diam, zero-lag saat tuas digerakkan)
"""
import math
import time

def smoothing_factor(t_e, cutoff):
    """Menghitung faktor pemulusan alpha berdasarkan cut-off frekuensi."""
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    """Pemulusan eksponensial standar."""
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    """
    1 Euro Filter (Low-Pass Filter Adaptif)
    Menghilangkan jitter (getaran) saat kursor diam, 
    dan menghilangkan lag (jeda) saat tuas digerakkan cepat.
    """
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=0.08, beta=1.2, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return x 
            
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)
        
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


def dist_3d(p1, p2):
    """
    Menghitung jarak Euclidean 3D antara dua landmark.
    Menggunakan koordinat 3D ternormalisasi jika tersedia (index 4, 5, 6),
    atau koordinat piksel 2D/3D sebagai fallback.
    """
    if len(p1) >= 7 and len(p2) >= 7:
        return math.sqrt((p1[4] - p2[4])**2 + (p1[5] - p2[5])**2 + (p1[6] - p2[6])**2)
    elif len(p1) >= 4 and len(p2) >= 4:
        return math.sqrt((p1[1] - p2[1])**2 + (p1[2] - p2[2])**2 + (p1[3] - p2[3])**2)
    return math.hypot(p1[1] - p2[1], p1[2] - p2[2])


class GestureRecognizer:
    def __init__(self):
        self.one_euro_filter = None
        
        # State Tracking
        self.is_currently_engaged = False
        
        # Hysteresis Schmitt Trigger Memory per-Jari (Melenyapkan 100% Boundary Flickering)
        self.finger_states = {
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }
        
        # Bi-directional Debouncing (Pencegah False Positive & False Drop)
        self.engage_frames_count = 0
        self.release_frames_count = 0
        self.REQUIRED_ENGAGE_FRAMES = 2   # Cukup 2 frame stabil untuk mengunci (Cepat & Anti-glitch)
        self.REQUIRED_RELEASE_FRAMES = 3  # Wajib 3 frame tangan terbuka untuk melepas rem

    def is_finger_extended_3d(self, lm_list, mcp_id, pip_id, tip_id, finger_key=None):
        """
        Menilai kelurusan jari menggunakan 3D Phalanx Extension Ratio & Hysteresis Schmitt Trigger:
        Ratio = dist_3d(Tip, MCP) / dist_3d(PIP, MCP)
        - Ambang Aktif (Lurus): Ratio >= 1.40
        - Ambang Lepas (Tekuk): Ratio < 1.25
        - Deadband selebar 0.15 melenyapkan 100% osilasi status jari (Anti-flickering).
        - Bersifat 100% Invarian terhadap kompresi perspektif 2D kamera saat tangan dimiringkan.
        """
        mcp = lm_list[mcp_id]
        pip = lm_list[pip_id]
        tip = lm_list[tip_id]
        
        d_tip_mcp = dist_3d(tip, mcp)
        d_pip_mcp = dist_3d(pip, mcp)
        
        if d_pip_mcp == 0:
            return False
            
        ratio = d_tip_mcp / d_pip_mcp
        
        # Schmitt Trigger State Hysteresis
        if finger_key and finger_key in self.finger_states:
            prev_state = self.finger_states[finger_key]
            if prev_state:
                # Sedang lurus: Hanya jatuh ke tekuk jika ratio turun di bawah 1.25
                new_state = (ratio >= 1.25)
            else:
                # Sedang tekuk: Hanya naik ke lurus jika ratio tembus di atas 1.40
                new_state = (ratio >= 1.40)
            self.finger_states[finger_key] = new_state
            return new_state
            
        return ratio >= 1.35

    def get_fingers_up(self, lm_list):
        """
        Mendeteksi jari-jari yang tegak lurus menggunakan kalkulasi 3D Phalanx Ratio.
        Mengembalikan [thumb, index, middle, ring, pinky] dalam biner (0 atau 1).
        """
        fingers = [0] # Jempol
        if len(lm_list) < 21:
            return fingers

        # Konfigurasi Sendi: (MCP, PIP, TIP, finger_key)
        finger_joints = [
            (5, 6, 8, 'index'),
            (9, 10, 12, 'middle'),
            (13, 14, 16, 'ring'),
            (17, 18, 20, 'pinky')
        ]

        for mcp, pip, tip, key in finger_joints:
            is_up = self.is_finger_extended_3d(lm_list, mcp, pip, tip, finger_key=key)
            fingers.append(1 if is_up else 0)
            
        return fingers

    def get_joystick_state(self, lm_list):
        """
        Menganalisis status tuas pergelangan tangan (is_engaged, smoothed_tilt).
        Bebas singularitas arcsin, linear 1:1, dan kebal dari noise.
        """
        if len(lm_list) < 21:
            self.reset_smoothing()
            return False, None

        # Evaluasi Status Jari berbasis 3D Phalanx & Hysteresis Schmitt Trigger
        is_index_up = self.is_finger_extended_3d(lm_list, 5, 6, 8, finger_key='index')
        is_middle_up = self.is_finger_extended_3d(lm_list, 9, 10, 12, finger_key='middle')
        is_ring_up = self.is_finger_extended_3d(lm_list, 13, 14, 16, finger_key='ring')
        
        # =======================================================
        # KALKULASI KEMIRINGAN LINEAR TERNORMALISASI 3D
        # =======================================================
        # Mengukur perbedaan vertikal antara ujung telunjuk dan pergelangan tangan,
        # dinormalisasi terhadap jarak tulang telapak tangan 3D (P0 ke P5).
        # Jarak 3D konstan mutlak, mencegah penyusutan skala saat pergelangan ditundukkan.
        wrist = lm_list[0]
        index_tip = lm_list[8]
        index_base = lm_list[5]

        dx_base = index_base[1] - wrist[1]
        dy_base = index_base[2] - wrist[2]
        dz_base = (index_base[3] - wrist[3]) if len(index_base) >= 4 else 0.0
        ref_dist_3d = math.sqrt(dx_base**2 + dy_base**2 + dz_base**2)
        if ref_dist_3d == 0:
            ref_dist_3d = 1.0

        # Kemiringan linear (skala persen terhadap panjang tulang telapak tangan 3D)
        raw_tilt = ((index_tip[2] - wrist[2]) / ref_dist_3d) * 100
        current_time = time.time()

        # =======================================================
        # 1 EURO FILTER: Adaptif Jitter & Lag Removal
        # =======================================================
        if self.one_euro_filter is None:
            self.one_euro_filter = OneEuroFilter(current_time, raw_tilt, min_cutoff=0.08, beta=1.2)
            smoothed_tilt = raw_tilt
        else:
            smoothed_tilt = self.one_euro_filter(current_time, raw_tilt)

        # =======================================================
        # BI-DIRECTIONAL DEBOUNCING & STATE MACHINE
        # =======================================================
        if not self.is_currently_engaged:
            # Kondisi ON: Menunjuk murni (Telunjuk lurus, Jari tengah & manis tertekuk santai)
            if is_index_up and not is_middle_up and not is_ring_up:
                self.engage_frames_count += 1
                if self.engage_frames_count >= self.REQUIRED_ENGAGE_FRAMES:
                    self.is_currently_engaged = True
                    self.engage_frames_count = 0
                    self.release_frames_count = 0
            else:
                self.engage_frames_count = 0
        else:
            # Kondisi OFF (Pengereman):
            # Tangan membuka (Jari tengah dan manis terentang lurus)
            if is_middle_up and is_ring_up:
                self.release_frames_count += 1
                if self.release_frames_count >= self.REQUIRED_RELEASE_FRAMES:
                    self.is_currently_engaged = False
                    self.release_frames_count = 0
                    self.engage_frames_count = 0
            else:
                # Retensi Stabil: Reset antrean pelepasan jika masih mempertahankan posisi
                self.release_frames_count = 0

        return self.is_currently_engaged, smoothed_tilt

    def reset_smoothing(self):
        """Mereset seluruh state filter dan debouncing buffer."""
        self.one_euro_filter = None
        self.is_currently_engaged = False
        self.engage_frames_count = 0
        self.release_frames_count = 0
        self.finger_states = {
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }
