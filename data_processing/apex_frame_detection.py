import cv2
import numpy as np
import math
from skimage.feature import local_binary_pattern

class ApexFrameSpotter:
    def __init__(self, t_window=61, sub_blocks=(6, 6), top_k_blocks=14, lbp_points=8, lbp_radius=1):
        """
        Apex Frame Spotter using 3D-FFT on LBP features.
        Based on "Joint Local and Global Information Learning With Single Apex Frame Detection for Micro-Expression Recognition"
        
        Args:
            t_window (int): Length of the sliding window T. (e.g., 61 for CASME II).
            sub_blocks (tuple): Division of face into blocks (rows, cols). Default (6, 6).
            top_k_blocks (int): Number of blocks with largest frequency to sum (N). Default 14.
            lbp_points (int): LBP points.
            lbp_radius (int): LBP radius.
        """
        self.t_window = t_window
        self.sub_blocks = sub_blocks
        self.top_k_blocks = top_k_blocks
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        
        # Threshold D0 is floor(T/2) based on the paper
        self.d0 = math.floor(t_window / 2)

    def _preprocess_sequence(self, frames):
        """
        Convert frames to LBP sequences.
        frames: list or array of (H, W, 3) or (H, W)
        Returns: (T, H, W) array of LBP features.
        """
        processed = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Extract LBP
            lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method='default')
            processed.append(lbp)
            
        return np.array(processed)

    def _divide_into_blocks(self, lbp_sequence):
        """
        Divide the (T, H, W) sequence into blocks of shape (NumBlocks, T, h_b, w_b).
        """
        T, H, W = lbp_sequence.shape
        r, c = self.sub_blocks
        h_step = H // r
        w_step = W // c
        
        blocks = []
        for i in range(r):
            for j in range(c):
                y_start = i * h_step
                y_end = (i + 1) * h_step
                x_start = j * w_step
                x_end = (j + 1) * w_step
                
                # If it's the last block, extend to the edge handling potential rounding
                if i == r - 1: y_end = H
                if j == c - 1: x_end = W
                
                sub_vol = lbp_sequence[:, y_start:y_end, x_start:x_end]
                blocks.append(sub_vol)
        return blocks

    def _compute_high_freq_amplitude(self, block_vol):
        """
        Compute the summed high-frequency amplitude for a single block volume (T, h, w).
        Uses 3D-FFT and HBF filter.
        """
        # FFT over Time, Height, Width
        if block_vol.size == 0:
            return 0.0
            
        freq_domain = np.fft.fftn(block_vol) 
        
        # It's easier to work with shifted frequencies (center at 0)
        freq_shifted = np.fft.fftshift(freq_domain)
        
        T, H, W = freq_shifted.shape
        ct, ch, cw = T//2, H//2, W//2
        
        # Grid of coordinates relative to center
        # Note: We want coordinates corresponding to frequency bins.
        # fftshift puts 0 freq at center.
        
        # Create grid indices
        # range: -center to +center (roughly)
        # Using meshgrid to interpret (u, v, q)
        # Paper Eq 2: sqrt(u^2 + v^2 + q^2) >= D0.
        
        # Coordinates
        q_range = np.arange(T) - ct
        u_range = np.arange(H) - ch
        v_range = np.arange(W) - cw
        
        # meshgrid(u, v, q) -> returns 3D arrays
        # Note the order in paper: F(u, v, q) -> corresponding to (x, y, z) or (H, W, T).
        # Usually standard FFT images are (H, W). Here 3D is (T, H, W) in our array.
        # Let's align:
        # Array shape: (T, H, W).
        # Dims: 0->Time (q), 1->Height (u), 2->Width (v)
        
        # Use 'indexing="ij"' to match dimensions 0, 1, 2
        Q, U, V = np.meshgrid(q_range, u_range, v_range, indexing='ij')
        
        dist = np.sqrt(U**2 + V**2 + Q**2)
        
        # High-pass filter mask
        mask = (dist >= self.d0)
        
        filtered = freq_shifted * mask
        
        # Amplitude is sum of absolute values
        # Eq 4: Sum of |G|
        amplitude = np.sum(np.abs(filtered))
        
        return amplitude

    def score_window(self, frames):
        """
        Compute the Apex Score (Amplitude A_i) for a specific window of frames.
        This corresponds to one 'interval' i in the paper.
        Args:
            frames: List or array of frames. Length should ideally be self.t_window.
        Returns:
            float: The amplitude score A_i.
        """
        # Preprocess
        lbp_seq = self._preprocess_sequence(frames)
        
        # Divide into blocks
        blocks = self._divide_into_blocks(lbp_seq)
        
        # Compute amplitude for each block
        block_amplitudes = []
        for blk in blocks:
            amp = self._compute_high_freq_amplitude(blk)
            block_amplitudes.append(amp)
            
        # Sort and take top N
        block_amplitudes.sort(reverse=True)
        top_N_sum = sum(block_amplitudes[:self.top_k_blocks])
        
        return top_N_sum

    def find_apex_in_short_video(self, video_path_or_frames, verbose=False):
        """
        Given a short video (e.g., a cropped ME clip), slide the window
        and find the single apex frame with the maximum score.
        Paper: "The intverval with maximum amplitude... The middle of the interval can be viewed as the apex frame."
        
        Returns:
            best_apex_index (int): Index of the apex frame in the video.
            max_score (float): The score of that window.
            all_scores (list): List of (frame_index, score) tuples.
        """
        frames = self._load_frames(video_path_or_frames)
            
        N = len(frames)
        if N < self.t_window:
            # If video is shorter than window, process the whole video as one window
            if verbose: print(f"Warning: Video length {N} < window {self.t_window}. Using full video.")
            score = self.score_window(frames)
            return N // 2, score, [(N//2, score)]

        scores = []
        # Slide window
        # Step size 1 for max precision
        for i in range(N - self.t_window + 1):
            window = frames[i : i + self.t_window]
            val = self.score_window(window)
            
            # The apex corresponding to this window is the middle frame
            apex_idx = i + self.t_window // 2
            scores.append((apex_idx, val))
            
        # Find best
        if not scores:
             # Fallback
             return N//2, 0, []

        best_i, best_val = max(scores, key=lambda x: x[1])
        return best_i, best_val, scores

    def find_apexes_in_long_video(self, video_path_or_frames, step_size=None):
        """
        For a long video, apply sliding window. Return multiple apex candidates.
        Users might want to look for local maxima above a threshold.
        
        Args:
            video_path_or_frames: Input.
            step_size: Stride for sliding window. Default t_window // 2.
        
        Returns:
            candidates: List of (frame_index, score).
        """
        frames = self._load_frames(video_path_or_frames)
            
        if step_size is None:
            step_size = max(1, self.t_window // 4)  # Overlap 75%
            
        N = len(frames)
        scores = []
        
        for i in range(0, N - self.t_window + 1, step_size):
            window = frames[i : i + self.t_window]
            val = self.score_window(window)
            mid_idx = i + self.t_window // 2
            scores.append((mid_idx, val))
            
        return scores

    def _load_frames(self, input_data):
        if isinstance(input_data, str):
            cap = cv2.VideoCapture(input_data)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            cap.release()
            return frames
        else:
            return input_data

if __name__ == "__main__":
    # Test script
    # Create random video data
    frames = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(100)]
    
    detector = ApexFrameSpotter(t_window=30, top_k_blocks=5)
    
    # 1. Test score specific window
    w_score = detector.score_window(frames[:30])
    print(f"Window Score: {w_score}")
    
    # 2. Test short video
    apex_idx, max_sc, _ = detector.find_apex_in_short_video(frames)
    print(f"Short Video Apex: Frame {apex_idx} with score {max_sc}")
    
    # 3. Test long video
    scores = detector.find_apexes_in_long_video(frames, step_size=5)
    print(f"Long Video: Computed {len(scores)} window scores.")
