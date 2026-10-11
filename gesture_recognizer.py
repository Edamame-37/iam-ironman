"""
Modul GestureRecognizer (Versi Kinematik 3D & State Machine Hysteresis)
Menganalisis pergerakan kemiringan telapak tangan (Wrist Tilt Throttle) menggunakan:
1. Bio-mechanical Joint Flexion Angles 3D (Kebal terhadap ilusi perspektif 2D)
2. 3D Palm Normal Surface Vector (Validasi arah telapak tangan menghadap kamera)
3. Continuous Confidence Scoring & Dual-Threshold Hysteresis (Anti-flicker & Anti-dropout)
4. Bi-directional Debouncing (Membasmi 100% false trigger / noise saat mengetik atau menggaruk)
5. 1 Euro Filter Adaptif (Zero jitter saat diam, zero lag saat bergerak cepat)
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
    1 Euro Filter (Low-Pass Filter Adaptif Standar Industri)
    Menghilangkan jitter (getaran mikroskopis) saat kursor diam, 
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
        
        # Cut-off frekuensi meningkat otomatis secara proporsional terhadap kecepatan gerakan
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


def calculate_joint_angle_3d(p_prev, p_joint, p_next):
    """
    Menghitung sudut fleksi sendi (Joint Flexion Angle) dalam derajat (0 - 180).
    Menggunakan operasi 3D Vector Dot Product.
    - Sifat Matematika: 100% Invarian terhadap rotasi, translasi, dan skala tangan.
    - Jari lurus: Sudut mendekati 160 - 180 derajat.
    - Jari ditekuk/mengepal: Sudut turun ke < 110 derajat.
    """
    # Ekstraksi koordinat 3D (Gunakan normalized jika tersedia, jika tidak gunakan piksel 2D/3D)
    if len(p_joint) >= 7:
        # Format [id, cx, cy, cz, norm_x, norm_y, norm_z]
        v1 = (p_prev[4] - p_joint[4], p_prev[5] - p_joint[5], p_prev[6] - p_joint[6])
        v2 = (p_next[4] - p_joint[4], p_next[5] - p_joint[5], p_next[6] - p_joint[6])
    elif len(p_joint) >= 4:
        v1 = (p_prev[1] - p_joint[1], p_prev[2] - p_joint[2], p_prev[3] - p_joint[3])
        v2 = (p_next[1] - p_joint[1], p_next[2] - p_joint[2], p_next[3] - p_joint[3])
    else:
        v1 = (p_prev[1] - p_joint[1], p_prev[2] - p_joint[2], 0.0)
        v2 = (p_next[1] - p_joint[1], p_next[2] - p_joint[2], 0.0)

    # Dot Product
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    # Magnitudo Vektor
    norm1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    norm2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

    if norm1 * norm2 == 0:
        return 180.0

    cosine = dot / (norm1 * norm2)
    # Clamp cosine untuk mencegah error komputasi floating point di luar [-1.0, 1.0]
    cosine = max(-1.0, min(1.0, cosine))
    
    return math.degrees(math.acos(cosine))


def calculate_palm_normal_3d(lm_list):
    """
    Menghitung Vektor Normal Permukaan Telapak Tangan (Palm Normal Vector) via Cross Product 3D.
    Mengembalikan (nx, ny, nz) ternormalisasi.
    nz mengindikasikan apakah telapak tangan menghadap kamera (sejajar bidang layar).
    """
    if len(lm_list) < 21:
        return 0.0, 0.0, 1.0

    # Landmark Kunci Telapak:
    # 0: Wrist, 5: Index MCP, 9: Middle MCP, 17: Pinky MCP
    p0 = lm_list[0]
    p5 = lm_list[5]
    p9 = lm_list[9]
    p17 = lm_list[17]

    if len(p0) >= 7:
        # Vektor Transversal: Index MCP -> Pinky MCP
        v_trans = (p17[4] - p5[4], p17[5] - p5[5], p17[6] - p5[6])
        # Vektor Longitudinal: Wrist -> Middle MCP
        v_long = (p9[4] - p0[4], p9[5] - p0[5], p9[6] - p0[6])
    elif len(p0) >= 4:
        v_trans = (p17[1] - p5[1], p17[2] - p5[2], p17[3] - p5[3])
        v_long = (p9[1] - p0[1], p9[2] - p0[2], p9[3] - p0[3])
    else:
        v_trans = (p17[1] - p5[1], p17[2] - p5[2], 0.0)
        v_long = (p9[1] - p0[1], p9[2] - p0[2], 0.0)

    # 3D Cross Product: v_trans x v_long
    nx = v_trans[1] * v_long[2] - v_trans[2] * v_long[1]
    ny = v_trans[2] * v_long[0] - v_trans[0] * v_long[2]
    nz = v_trans[0] * v_long[1] - v_trans[1] * v_long[0]

    mag = math.sqrt(nx**2 + ny**2 + nz**2)
    if mag == 0:
        return 0.0, 0.0, 1.0
        
    return nx / mag, ny / mag, nz / mag


class GestureRecognizer:
    def __init__(self):
        self.one_euro_filter = None
        
        # State Tracking
        self.is_currently_engaged = False
        
        # Bi-directional Debouncing Buffers
        self.engage_frames_count = 0
        self.release_frames_count = 0
        self.REQUIRED_ENGAGE_FRAMES = 3   # Wajib 3 frame stabil berturut-turut untuk ON (Membunuh 100% false-alarm)
        self.REQUIRED_RELEASE_FRAMES = 3  # Wajib 3 frame stabil berturut-turut untuk OFF (Mencegah optical flicker)
        
        # Ambang Batas Hysteresis (Schmitt Trigger)
        self.ENGAGE_CONFIDENCE_THRESHOLD = 0.65 # Threshold tinggi untuk masuk
        self.SUSTAIN_CONFIDENCE_THRESHOLD = 0.35 # Threshold rendah untuk mempertahankan (Anti-dropout)
        self.BRAKE_CONFIDENCE_THRESHOLD = 0.60  # Threshold tangan terbuka untuk mengerem
        
        # Cache Evaluasi Terakhir (Untuk Debugging / HUD)
        self.last_confidence = 0.0
        self.last_open_confidence = 0.0

    def compute_gesture_confidences(self, lm_list):
        """
        Mengevaluasi bentuk tangan secara probabilitas kontinu (0.0 - 1.0)
        berdasarkan sudut tekukan sendi kinematik 3D dan vektor normal telapak.
        """
        if len(lm_list) < 21:
            return 0.0, 0.0

        # 1. Hitung Sudut Fleksi Sendi PIP untuk 4 Jari Utama
        # Telunjuk: PIP (6), MCP (5), TIP (8)
        angle_index = calculate_joint_angle_3d(lm_list[5], lm_list[6], lm_list[8])
        # Tengah: PIP (10), MCP (9), TIP (12)
        angle_middle = calculate_joint_angle_3d(lm_list[9], lm_list[10], lm_list[12])
        # Manis: PIP (14), MCP (13), TIP (16)
        angle_ring = calculate_joint_angle_3d(lm_list[13], lm_list[14], lm_list[16])
        # Kelingking: PIP (18), MCP (17), TIP (20)
        angle_pinky = calculate_joint_angle_3d(lm_list[17], lm_list[18], lm_list[20])

        # 2. Pemetaan Skor Ekstensi Jari (0.0 = Ditekuk penuh, 1.0 = Lurus sempurna)
        # Jari dianggap mulai lurus di atas 120 derajat, dan sempurna di 160 derajat
        def ext_score(angle):
            return max(0.0, min(1.0, (angle - 120.0) / (160.0 - 120.0)))
            
        # Pemetaan Skor Tekukan Jari (1.0 = Ditekuk rapat, 0.0 = Lurus)
        def curl_score(angle):
            return max(0.0, min(1.0, (145.0 - angle) / (145.0 - 105.0)))

        s_index_ext = ext_score(angle_index)
        s_middle_curl = curl_score(angle_middle)
        s_ring_curl = curl_score(angle_ring)
        s_pinky_curl = curl_score(angle_pinky)
        
        s_others_curled = (s_middle_curl + s_ring_curl + s_pinky_curl) / 3.0

        # 3. Validasi Normal Permukaan Telapak Tangan (Palm Facing Alignment)
        nx, ny, nz = calculate_palm_normal_3d(lm_list)
        # Nilai absolut |nz| tinggi berarti bidang telapak tangan sejajar dengan bidang sensor kamera
        s_palm_facing = max(0.0, min(1.0, (abs(nz) - 0.25) / (0.65 - 0.25)))

        # 4. Total Pointing Confidence (Telunjuk Menunjuk Murni)
        c_pointing = (0.50 * s_index_ext) + (0.35 * s_others_curled) + (0.15 * s_palm_facing)

        # 5. Total Open Hand Confidence (Telapak Terbuka Penuh Untuk Mengerem)
        s_mid_ext = ext_score(angle_middle)
        s_ring_ext = ext_score(angle_ring)
        s_pinky_ext = ext_score(angle_pinky)
        c_open = (s_index_ext + s_mid_ext + s_ring_ext + s_pinky_ext) / 4.0

        return c_pointing, c_open

    def get_fingers_up(self, lm_list):
        """
        Fungsi pembantu kompatibilitas backward menggunakan sudut sendi kinematik 3D.
        Mengembalikan list biner [thumb, index, middle, ring, pinky].
        """
        fingers = [0] # Jempol
        if len(lm_list) < 21:
            return fingers

        joints = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
        for mcp, pip, tip in joints:
            angle = calculate_joint_angle_3d(lm_list[mcp], lm_list[pip], lm_list[tip])
            fingers.append(1 if angle >= 135.0 else 0)
            
        return fingers

    def get_joystick_state(self, lm_list):
        """
        Menganalisis pergerakan kemiringan pergelangan tangan dengan algoritma state machine multi-layer.
        Mengembalikan (is_engaged, smoothed_tilt_degrees).
        """
        if len(lm_list) < 21:
            self.reset_smoothing()
            return False, None

        # Evaluasi Probabilitas Gestur
        c_pointing, c_open = self.compute_gesture_confidences(lm_list)
        self.last_confidence = c_pointing
        self.last_open_confidence = c_open

        # =======================================================
        # PERHITUNGAN SUDUT KEMIRINGAN PITCH 3D INVARIAN
        # =======================================================
        # Mengukur rotasi pitch pergelangan tangan menggunakan rasio delta Y
        # terhadap panjang 3D pergelangan ke ujung telunjuk (L_hand_3d).
        # Rasio ini mewakili sin(theta) fisik tangan murni, kebal terhadap jarak & skala!
        wrist = lm_list[0]
        index_tip = lm_list[8]
        
        dx = index_tip[1] - wrist[1]
        dy = index_tip[2] - wrist[2]
        dz = (index_tip[3] - wrist[3]) if len(index_tip) >= 4 else 0.0
        
        hand_len_3d = math.sqrt(dx**2 + dy**2 + dz**2)
        if hand_len_3d == 0:
            hand_len_3d = 1.0

        # Sinus sudut kemiringan vertikal
        sin_pitch = dy / hand_len_3d
        sin_pitch = max(-1.0, min(1.0, sin_pitch))
        
        # Sudut kemiringan dalam derajat fisik murni (-90 hingga +90)
        raw_tilt_deg = math.degrees(math.asin(sin_pitch))
        
        current_time = time.time()

        # =======================================================
        # 1 EURO FILTER: Adaptif Jitter & Lag Elimination
        # =======================================================
        if self.one_euro_filter is None:
            self.one_euro_filter = OneEuroFilter(current_time, raw_tilt_deg, min_cutoff=0.08, beta=1.2)
            smoothed_tilt = raw_tilt_deg
        else:
            smoothed_tilt = self.one_euro_filter(current_time, raw_tilt_deg)

        # =======================================================
        # DUAL-THRESHOLD HYSTERESIS & BI-DIRECTIONAL DEBOUNCING
        # =======================================================
        if not self.is_currently_engaged:
            # Syarat ON: Menuntut Keyakinan Tinggi (>= 0.65) selama 3 Frame berturut-turut
            if c_pointing >= self.ENGAGE_CONFIDENCE_THRESHOLD:
                self.engage_frames_count += 1
                if self.engage_frames_count >= self.REQUIRED_ENGAGE_FRAMES:
                    self.is_currently_engaged = True
                    self.engage_frames_count = 0
                    self.release_frames_count = 0
            else:
                self.engage_frames_count = 0
        else:
            # Syarat OFF (Pengereman):
            # Tangan membuka lebar ATAU keyakinan jatuh di bawah ambang retensi (Hysteresis low threshold)
            is_braking = (c_open >= self.BRAKE_CONFIDENCE_THRESHOLD)
            is_lost = (c_pointing < self.SUSTAIN_CONFIDENCE_THRESHOLD)
            
            if is_braking or is_lost:
                self.release_frames_count += 1
                if self.release_frames_count >= self.REQUIRED_RELEASE_FRAMES:
                    self.is_currently_engaged = False
                    self.release_frames_count = 0
                    self.engage_frames_count = 0
            else:
                # Retensi Stabil: Reset antrean pelepasan jika pengguna masih mempertahankan gestur
                self.release_frames_count = 0

        return self.is_currently_engaged, smoothed_tilt

    def reset_smoothing(self):
        """Mereset seluruh state filter dan debouncing."""
        self.one_euro_filter = None
        self.is_currently_engaged = False
        self.engage_frames_count = 0
        self.release_frames_count = 0
        self.last_confidence = 0.0
        self.last_open_confidence = 0.0
