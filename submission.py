# %% [markdown]
# # Seele — Kaggle ConnectX Agent
# 
# Single-file submission. Cells mirror the logical sections of .

# %%
"""Kaggle ConnectX agent.

Single-file submission for the Kaggle ConnectX simulation competition.

For the standard 7-wide, 6-tall, K = 4 board, the agent tries three things in
order:

1.  When playing Player 1 (Red), use 2swap's WeakC4 weak-solution book. The
    book is an ~8.6 K-node graph compressed into ``_BOOK_BLOB`` below. Each
    Red-to-move position resolves to either a forced move (internal node) or
    a "steady-state" priority grid (leaf node) that picks a perfect move
    using only the local board state. See ``WeakC4/explanation/index.html``.
2.  Otherwise (or when the book misses), run a negamax + alpha-beta search
    with a transposition table, time-bounded iterative deepening, and the
    forced-move static analysis from the C++ solver in ``src/solver/``.
3.  For any non-default board configuration, run a generic depth-limited
    minimax with a window-counting heuristic.
"""

import base64
import pickle
import time
import zlib


# %% [markdown]
# ## Bitboard Primitives

# %%
# =============================================================================
# Bitboard Primitives - mã hóa bàn cờ dạng số nguyên (bitmask)
#
# CÁCH BIỂU DIỄN BITBOARD:
# Mỗi cột dùng BH+1 = 7 bit (6 ô thực + 1 bit sentinel ở trên cùng).
# Bit thứ 0 của cột c = ô dưới cùng, bit thứ 5 = ô trên cùng.
#
#   Cột 0  | Cột 1  | ... | Cột 6
#   bit 6  | bit 13 |     | bit 48  <- sentinel (luôn = 0, ngăn bit tràn sang cột khác)
#   bit 5  | bit 12 |     | bit 47  <- hàng 5 (trên cùng)
#   bit 4  | bit 11 |     | bit 46
#   bit 3  | bit 10 |     | bit 45
#   bit 2  | bit 9  |     | bit 44
#   bit 1  | bit 8  |     | bit 43
#   bit 0  | bit 7  |     | bit 42  <- hàng 0 (dưới cùng)
#
# Ví dụ: b0 = 0b1 nghĩa là có quân ở cột 0, hàng 0 (góc dưới trái).
# =============================================================================

BW = 7       # số cột (Board Width)
BH = 6       # số hàng (Board Height)
BH1 = BH + 1 # = 7, số bit mỗi cột (gồm cả sentinel)
NCELLS = BW * BH  # = 42, tổng số ô trên bàn cờ


def _ones(n):
    """Tạo bitmask gồm n bit 1: _ones(4) = 0b1111 = 15."""
    return (1 << n) - 1


# --- Các mask nền tảng ---

# FIRST_COLUMN: 6 bit thấp nhất = vị trí các ô của cột 0 (không tính sentinel)
FIRST_COLUMN = _ones(BH)  # = 0b111111

# BOTTOM_ROW: bit dưới cùng của mỗi cột
#   = 1<<0 | 1<<7 | 1<<14 | 1<<21 | 1<<28 | 1<<35 | 1<<42
# Công thức: lấy chuỗi 49 bit 1 chia cho chuỗi 7 bit 1 => mỗi cột chỉ còn bit thấp nhất
BOTTOM_ROW = _ones(BH1 * BW) // _ones(BH1)

# COLUMN_HEADERS: vị trí 7 bit sentinel (bit thứ BH = bit thứ 6 của mỗi cột)
COLUMN_HEADERS = BOTTOM_ROW << BH

# VALID_CELLS: 42 ô hợp lệ có thể chơi (= các ô từ hàng 0 đến hàng 5, không tính sentinel)
VALID_CELLS = COLUMN_HEADERS - BOTTOM_ROW

# Điểm thắng và vô cực cho thuật toán alpha-beta
WIN = 10000
INF = 100000

# Thứ tự ưu tiên cột khi tìm kiếm: bắt đầu từ giữa (cột 3) rồi ra ngoài dần
# Quân ở giữa mạnh hơn nên thử trước để tăng tốc cắt tỉa alpha-beta
_CENTER_ORDER = (3, 4, 2, 5, 1, 6, 0)

# --- Mask hàng chẵn/lẻ dùng cho heuristic đánh giá ---
#
# Lý thuyết Connect 4 (Even/Odd Rule):
#   - Player 1 (Red) được lợi khi có đe dọa ở hàng lẻ (hàng 1,3,5 tính từ dưới - 1-indexed)
#     = bit positions 0, 2, 4 trong mỗi cột
#   - Player 2 (Yellow) được lợi khi có đe dọa ở hàng chẵn (hàng 2,4,6 - 1-indexed)
#     = bit positions 1, 3, 5 trong mỗi cột
#
# Nguyên nhân: trong endgame, ai kiểm soát các hàng này sẽ tạo được "zugzwang"
# (buộc đối thủ phải tự tạo cơ hội thua).
_ODD_ROW_MASK  = sum(1 << (r + c * BH1) for c in range(BW) for r in (0, 2, 4))
_EVEN_ROW_MASK = sum(1 << (r + c * BH1) for c in range(BW) for r in (1, 3, 5))


def _has_won(b):
    """Trả về True nếu bitboard b chứa 4 quân liên tiếp theo bất kỳ hướng nào.

    Kỹ thuật bit: nếu pairs = b & (b << s) thì bit 1 tại vị trí x có nghĩa là
    x và x+s đều có quân. Tiếp tục: pairs & (pairs << 2s) cho ra vị trí có 4 quân liên tiếp.
    """
    # Kiểm tra dọc: 4 quân liên tiếp theo cột (stride = 1 bit)
    pairs = b & (b << 1)
    if pairs & (pairs << 2):
        return True

    # Kiểm tra ngang: 4 quân liên tiếp theo hàng (stride = BH1 = 7 bit = 1 cột)
    pairs = b & (b << BH1)
    if pairs & (pairs << (2 * BH1)):
        return True

    # Kiểm tra chéo lên phải (stride = BH = 6 bit)
    pairs = b & (b << BH)
    if pairs & (pairs << (2 * BH)):
        return True

    # Kiểm tra chéo xuống phải (stride = BH + 2 = 8 bit)
    pairs = b & (b << (BH + 2))
    if pairs & (pairs << (2 * (BH + 2))):
        return True

    return False


def _find_threats(b):
    """Tìm các ô mà nếu đặt quân vào thì tạo 4 liên tiếp cho b.

    Nhận diện 4 kiểu đe dọa (X = quân có, _ = ô đe dọa cần tìm):
      _XXX  <-  ô bên trái/dưới 3 quân liên tiếp
      XXX_  ->  ô bên phải/trên 3 quân liên tiếp
      X_XX  ->  ô lỗ ở vị trí thứ 2
      XX_X  ->  ô lỗ ở vị trí thứ 3

    Kết quả cần lọc thêm bằng cách AND với (~opponent & VALID_CELLS).
    """
    threats = 0

    # Dọc: chỉ có kiểu XXX_ (ô phía trên 3 quân thẳng đứng)
    pairs  = b & (b << 1)
    triple = pairs & (pairs << 1)
    threats |= triple << 1   # ô ngay trên 3 quân dọc

    # Ngang + 2 đường chéo: hỗ trợ cả 4 kiểu đe dọa
    for stride in (BH1, BH, BH + 2):
        pairs  = b & (b << stride)
        triple = pairs & (pairs << stride)

        # Kiểu X_XX: ô bên trái cặp đôi (bên phải quân lẻ đầu)
        threats |= (b >> stride) & (pairs << stride)

        # Kiểu XX_X: ô bên phải cặp đôi (bên trái quân lẻ cuối)
        threats |= (b << stride) & (pairs >> (2 * stride))

        # Kiểu XXX_: ô ngay sau 3 quân liên tiếp
        threats |= triple << stride

        # Kiểu _XXX: ô ngay trước 3 quân liên tiếp
        threats |= triple >> (3 * stride)

    return threats & VALID_CELLS


def _col_of_mask(mask):
    """Trả về chỉ số cột của bit thấp nhất trong mask (mask phải != 0)."""
    lowest_bit_pos = (mask & -mask).bit_length() - 1
    return lowest_bit_pos // BH1


def _col_mask(col, occ):
    """Trả về bit của ô chơi được tiếp theo trong cột col (0 nếu cột đầy).

    Cộng BOTTOM_ROW vào occ sẽ 'đẩy' lên 1 ô trong mỗi cột.
    AND với vị trí của cột col để lấy ô trống thấp nhất trong cột đó.
    """
    return (occ + BOTTOM_ROW) & (FIRST_COLUMN << (BH1 * col))


def _popcount(x):
    """Đếm số bit 1 trong x. Dùng int.bit_count() có sẵn từ Python 3.10."""
    return x.bit_count()


# Mask từng cột riêng lẻ (dùng cho hàm _mirror)
_COL_MASKS = tuple(_ones(BH1) << (c * BH1) for c in range(BW))


def _mirror(b):
    """Lật bitboard b theo chiều ngang: cột 0 <-> cột 6, cột 1 <-> cột 5, ...

    Connect 4 có tính đối xứng trái-phải: một vị trí và bản gương của nó
    có cùng giá trị trò chơi. Dùng điều này để giảm ~50% kích thước TT.
    """
    result = 0
    for col in range(BW):
        col_bits = (b >> (col * BH1)) & _ones(BH1)
        result |= col_bits << ((BW - 1 - col) * BH1)
    return result


def _canonical(b0, b1):
    """Chuẩn hóa cặp (b0, b1) để dùng làm khóa trong bảng chuyển vị.

    Chọn dạng nhỏ hơn trong (b0, b1) và (mirror(b0), mirror(b1)).
    Trả về: (b0_chuẩn, b1_chuẩn, đã_lật_gương)
    Nếu đã_lật_gương=True, cột tốt nhất đọc từ TT cần đổi: col -> BW-1-col.
    """
    mirrored_b0 = _mirror(b0)
    if mirrored_b0 == b0:
        # Vị trí đối xứng với chính nó, không cần lật
        return b0, b1, False
    mirrored_b1 = _mirror(b1)
    if (mirrored_b0, mirrored_b1) < (b0, b1):
        return mirrored_b0, mirrored_b1, True
    return b0, b1, False


def _threat_col_count(threats):
    """Đếm số cột có ít nhất 1 bit đe dọa. Dùng để phát hiện fork (2+ cột đe dọa cùng lúc)."""
    count = 0
    for c in range(BW):
        if threats & (FIRST_COLUMN << (c * BH1)):
            count += 1
    return count


def _eval(b0, b1, p_threats, o_threats):
    """Heuristic đánh giá thế cờ cho bên đang đi (b0).

    Cải tiến:
    - Playable threat weighting: đe dọa chơi được ngay (top of stack) > đe dọa bị chôn vùi
    - Fork detection: threats ở 2+ cột cùng lúc = đối thủ không thể chặn hết
    - Even/odd rule: Player 1 lợi hàng lẻ (bits 0,2,4), Player 2 lợi hàng chẵn (bits 1,3,5)
    """
    occupied = b0 | b1
    playable = (occupied + BOTTOM_ROW) & VALID_CELLS

    useful_mine = p_threats & ~(o_threats << 1)
    useful_opp  = o_threats & ~(p_threats << 1)

    is_p1 = b0.bit_count() == b1.bit_count()
    good_rows_mine = _ODD_ROW_MASK  if is_p1 else _EVEN_ROW_MASK
    good_rows_opp  = _EVEN_ROW_MASK if is_p1 else _ODD_ROW_MASK

    # Threats playable ngay lập tức (ô trên cùng cột hiện tại)
    p_playable = _popcount(p_threats & playable)
    o_playable = _popcount(o_threats & playable)

    # Fork: threats ở 2+ cột = đối thủ chỉ chặn được 1
    p_fork_bonus = max(0, _threat_col_count(p_threats) - 1)
    o_fork_bonus = max(0, _threat_col_count(o_threats) - 1)

    score  = 12 * (p_playable - o_playable)                    # immediate threats: quan trọng nhất
    score += 6  * (_popcount(useful_mine & good_rows_mine) - _popcount(useful_opp & good_rows_opp))
    score += 2  * (_popcount(useful_mine) - _popcount(useful_opp))
    score +=      _popcount(p_threats) - _popcount(o_threats)
    score += 15 * (p_fork_bonus - o_fork_bonus)                # fork: rất nguy hiểm

    center = FIRST_COLUMN << (BH1 * (BW // 2))
    score += _popcount(b0 & center) - _popcount(b1 & center)
    return score


# %% [markdown]
# ## Negamax Alpha-Beta Search

# %%
# =============================================================================
# Tìm kiếm Negamax với PVS, LMR, Aspiration Windows, History Heuristic, TT
# =============================================================================
#
# NEGAMAX: điểm(bên A) = -điểm(bên B) => 1 hàm duy nhất thay vì max/min riêng.
#
# ALPHA-BETA: cắt nhánh không ảnh hưởng kết quả.
#   alpha = điểm tốt nhất ta đã đảm bảo được
#   beta  = điểm tốt nhất đối thủ đã đảm bảo được
#   alpha >= beta => dừng duyệt (beta cutoff)
#
# PVS (Principal Variation Search):
#   - Nước đầu tiên (tốt nhất theo TT/killer): tìm cửa sổ đầy đủ [-beta, -alpha]
#   - Các nước sau: dùng cửa sổ rỗng [-alpha-1, -alpha] (nhanh hơn)
#   - Nếu cửa sổ rỗng thất bại cao (score > alpha): re-search cửa sổ đầy đủ
#
# LMR (Late Move Reduction):
#   - Giảm depth-1 cho các nước đi muộn (move_num >= 2) trong cửa sổ rỗng PVS
#   - Nếu kết quả vẫn > alpha: re-search với depth đầy đủ
#
# ASPIRATION WINDOWS:
#   - Bắt đầu mỗi depth mới với cửa sổ hẹp (prev_score ± delta)
#   - Nếu thất bại: mở rộng cửa sổ và tìm lại
#   - Giảm số node cần duyệt ở mỗi vòng iterative deepening
#
# HISTORY HEURISTIC:
#   - Lưu cột nào thường gây ra beta-cutoff => thử trước trong move ordering
#   - Bổ sung cho killer moves, cải thiện chất lượng sắp xếp nước đi
#
# TRANSPOSITION TABLE (TT):
#   - Cache kết quả tìm kiếm, dùng lại giữa các lượt chơi
#   - Mỗi entry: (độ_sâu, loại_biên, điểm, cột_tốt_nhất)
#   - Loại biên: 0=EXACT, 1=LOWER, 2=UPPER
# =============================================================================

_GLOBAL_TT = {}
_TT_SIZE_CAP = 1_500_000   # 1.5M entries ~ 300MB; an toàn hơn 3M trên Kaggle (GC lag)

# Cửa sổ khởi tạo cho Aspiration Windows
_ASPIRATION_DELTA = 50


class _Searcher:
    __slots__ = ("deadline", "tt", "nodes", "aborted", "killers", "history", "counter_move")

    def __init__(self, deadline, tt):
        self.deadline = deadline  # thời điểm phải dừng (time.monotonic())
        self.tt = tt              # bảng chuyển vị (dict dùng chung)
        self.nodes = 0            # đếm số node đã xét
        self.aborted = False      # True khi hết thời gian
        # Killer moves: 2 slot/mức, thử trước để tăng tỷ lệ cắt tỉa
        self.killers = [[-1, -1] for _ in range(NCELLS + 2)]
        # History heuristic: lịch sử cột nào thường gây beta-cutoff
        self.history = [0] * BW
        # Counter move heuristic: best response khi đối thủ vừa đi cột last_col
        self.counter_move = [-1] * BW

    def negamax(self, my_board, opp_board, moves_made, alpha, beta, depth, last_col=-1):
        """Tìm kiếm negamax với PVS + LMR + TT + killer + history.

        my_board:   bitboard của bên đang đi
        opp_board:  bitboard của đối thủ
        moves_made: tổng số nước đã đi (để tính điểm thắng chính xác)
        alpha, beta: cửa sổ alpha-beta
        depth:      số lớp tìm kiếm còn lại
        Trả về: điểm từ góc nhìn của bên đang đi (dương = tốt cho ta)
        """
        self.nodes += 1
        # Kiểm tra thời gian mỗi 4096 node
        if (self.nodes & 0xFFF) == 0 and time.monotonic() >= self.deadline:
            self.aborted = True
            return 0

        # Đối thủ vừa thắng?
        if _has_won(opp_board):
            return -(WIN - moves_made)

        # Bàn cờ đầy = hòa
        if moves_made >= NCELLS:
            return 0

        occupied = my_board | opp_board
        playable_now = (occupied + BOTTOM_ROW) & VALID_CELLS

        # Thắng ngay?
        my_threats = _find_threats(my_board) & ~opp_board
        win_now = my_threats & playable_now
        if win_now:
            return WIN - moves_made - 1

        opp_threats = _find_threats(opp_board) & ~my_board
        opp_wins_now = opp_threats & playable_now
        safe_moves = playable_now & ~(opp_threats >> 1)

        # Thua bắt buộc?
        if safe_moves == 0:
            return -(WIN - moves_made - 2)
        if opp_wins_now:
            if opp_wins_now & (opp_wins_now - 1):
                return -(WIN - moves_made - 2)
            if not (opp_wins_now & safe_moves):
                return -(WIN - moves_made - 2)
            # Chỉ 1 nước chặn bắt buộc, không tiêu depth
            return -self.negamax(opp_board, my_board | opp_wins_now, moves_made + 1, -beta, -alpha, depth, _col_of_mask(opp_wins_now))

        if not (safe_moves & (safe_moves - 1)):
            # Chỉ 1 nước an toàn duy nhất, không tiêu depth
            return -self.negamax(opp_board, my_board | safe_moves, moves_made + 1, -beta, -alpha, depth, _col_of_mask(safe_moves))

        if depth == 0:
            return _eval(my_board, opp_board, my_threats, opp_threats)

        # === Tra cứu bảng chuyển vị ===
        canonical_b0, canonical_b1, mirrored = _canonical(my_board, opp_board)
        tt_key = (canonical_b0, canonical_b1)
        tt_entry = self.tt.get(tt_key)
        tt_move = -1
        if tt_entry is not None:
            tt_depth, bound_type, tt_score, tt_best_col = tt_entry
            if tt_depth >= depth:
                if bound_type == 0:
                    return tt_score
                if bound_type == 1:
                    if tt_score >= beta:
                        return tt_score
                    if tt_score > alpha:
                        alpha = tt_score
                else:
                    if tt_score <= alpha:
                        return tt_score
                    if tt_score < beta:
                        beta = tt_score
                if alpha >= beta:
                    return tt_score
            tt_move = tt_best_col if not mirrored else (BW - 1 - tt_best_col if tt_best_col >= 0 else -1)

        # === Sắp xếp nước đi: TT -> Killer -> History -> Center-out ===
        candidates = []
        for col in _CENTER_ORDER:
            move_bit = _col_mask(col, occupied)
            if move_bit and (move_bit & safe_moves):
                candidates.append((col, move_bit))

        front = 0
        # TT move lên đầu
        if tt_move >= 0:
            for i in range(front, len(candidates)):
                if candidates[i][0] == tt_move:
                    if i > front:
                        candidates.insert(front, candidates.pop(i))
                    front += 1
                    break
        # Killer moves lên sau TT
        for killer_col in self.killers[depth]:
            if killer_col < 0 or killer_col == tt_move:
                continue
            for i in range(front, len(candidates)):
                if candidates[i][0] == killer_col:
                    if i > front:
                        candidates.insert(front, candidates.pop(i))
                    front += 1
                    break
        # Counter move: best response đến nước trước của đối thủ
        if last_col >= 0:
            cm = self.counter_move[last_col]
            if cm >= 0 and cm != tt_move and cm not in self.killers[depth]:
                for i in range(front, len(candidates)):
                    if candidates[i][0] == cm:
                        if i > front:
                            candidates.insert(front, candidates.pop(i))
                        front += 1
                        break
        # Sắp xếp phần còn lại theo history heuristic
        if front < len(candidates) - 1:
            tail = candidates[front:]
            tail.sort(key=lambda x: -self.history[x[0]])
            candidates[front:] = tail

        # === Duyệt nước đi với PVS + LMR ===
        best_score = -INF
        best_col = candidates[0][0]
        original_alpha = alpha

        for move_num, (col, move_bit) in enumerate(candidates):
            new_board = my_board | move_bit

            # Dynamic LMR: reduction tăng theo move_num và depth
            # move_num 2-3: -1 | 4-5: -2 | 6+: -3 | depth>=6: thêm -1
            reduction = 0
            if move_num >= 2:
                reduction = (1 + (move_num >= 4) + (move_num >= 6) + (depth >= 6))
                reduction = min(reduction, depth - 1)  # không reduce xuống depth < 0

            if move_num == 0:
                # Nước đầu tiên (tốt nhất): tìm cửa sổ đầy đủ, không giảm depth
                child_score = -self.negamax(opp_board, new_board, moves_made + 1, -beta, -alpha, depth - 1, col)
            else:
                # PVS: thử cửa sổ rỗng trước (với LMR nếu có)
                child_score = -self.negamax(
                    opp_board, new_board, moves_made + 1,
                    -alpha - 1, -alpha, depth - 1 - reduction, col
                )
                # Nếu cửa sổ rỗng thất bại cao (score > alpha): re-search đầy đủ
                if not self.aborted and child_score > alpha:
                    child_score = -self.negamax(
                        opp_board, new_board, moves_made + 1,
                        -beta, -alpha, depth - 1, col
                    )

            if self.aborted:
                return 0
            if child_score > best_score:
                best_score = child_score
                best_col = col
            if child_score > alpha:
                alpha = child_score
            if alpha >= beta:
                # Beta cutoff: cập nhật killer, history và counter move
                k = self.killers[depth]
                if k[0] != col:
                    k[1] = k[0]
                    k[0] = col
                self.history[col] += 1 << min(depth, 20)
                if last_col >= 0:
                    self.counter_move[last_col] = col
                break

        # === Lưu vào bảng chuyển vị ===
        if best_score <= original_alpha:
            bound_type = 2   # UPPER
        elif best_score >= beta:
            bound_type = 1   # LOWER
        else:
            bound_type = 0   # EXACT
        stored_col = (BW - 1 - best_col) if mirrored else best_col
        self.tt[tt_key] = (depth, bound_type, best_score, stored_col)
        return best_score


def _solve_default(b0, b1, mp, deadline):
    """Iterative deepening với Aspiration Windows trên bàn cờ chuẩn 7x6 / K=4.

    Aspiration Windows: mỗi depth bắt đầu với cửa sổ hẹp (prev_score ± delta).
    Nếu thất bại (fail-low hoặc fail-high): mở rộng cửa sổ và tìm lại.
    Kết quả: cần ít node hơn so với full-window, đạt depth sâu hơn trong cùng thời gian.
    """
    occupied = b0 | b1
    playable_now = (occupied + BOTTOM_ROW) & VALID_CELLS

    # --- Xử lý tĩnh: không cần tìm kiếm ---
    my_threats = _find_threats(b0) & ~b1
    win_now = my_threats & playable_now
    if win_now:
        return _col_of_mask(win_now)

    opp_threats = _find_threats(b1) & ~b0
    opp_wins = opp_threats & playable_now
    if opp_wins:
        return _col_of_mask(opp_wins & -opp_wins)

    safe_moves = playable_now & ~(opp_threats >> 1)
    if safe_moves == 0:
        return _col_of_mask(playable_now & -playable_now)
    if not (safe_moves & (safe_moves - 1)):
        return _col_of_mask(safe_moves)

    # --- Thiết lập tìm kiếm ---
    if len(_GLOBAL_TT) > _TT_SIZE_CAP:
        _GLOBAL_TT.clear()
    searcher = _Searcher(deadline, _GLOBAL_TT)

    candidates = []
    for col in _CENTER_ORDER:
        move_bit = _col_mask(col, occupied)
        if move_bit and (move_bit & safe_moves):
            candidates.append((col, move_bit))

    best_col = candidates[0][0]
    prev_best_col = -1
    prev_score = None      # điểm tốt nhất của depth trước, làm trung tâm aspiration
    max_depth = NCELLS - mp

    # --- Iterative deepening với Aspiration Windows ---
    for depth in range(2, max_depth + 1):
        if time.monotonic() >= deadline:
            break

        # Đưa nước tốt nhất của depth trước lên đầu (move ordering)
        if prev_best_col >= 0:
            for i, (c, _) in enumerate(candidates):
                if c == prev_best_col:
                    if i > 0:
                        candidates.insert(0, candidates.pop(i))
                    break

        # Thiết lập cửa sổ aspiration
        # Không dùng aspiration khi gần thắng/thua (score xa 0) vì dễ fail
        if prev_score is not None and abs(prev_score) < WIN - NCELLS:
            a_lo = prev_score - _ASPIRATION_DELTA
            a_hi = prev_score + _ASPIRATION_DELTA
        else:
            a_lo, a_hi = -INF, INF

        round_aborted = False
        round_score = None
        round_col = candidates[0][0]

        # Vòng lặp aspiration: retry nếu score ngoài cửa sổ
        while not round_aborted:
            searcher.aborted = False
            curr_best = a_lo - 1   # khởi tạo dưới lower bound để phát hiện fail-low
            curr_col = candidates[0][0]

            for col, move_bit in candidates:
                # Truyền -a_hi và -max(a_lo, curr_best) cho child
                child_alpha = max(a_lo, curr_best)
                score = -searcher.negamax(b1, b0 | move_bit, mp + 1, -a_hi, -child_alpha, depth - 1, col)
                if searcher.aborted:
                    round_aborted = True
                    break
                if score > curr_best:
                    curr_best = score
                    curr_col = col

            if round_aborted:
                break

            # Kiểm tra kết quả so với cửa sổ
            if curr_best <= a_lo and a_lo > -INF:
                # Fail-low: thực tế điểm thấp hơn kỳ vọng => mở rộng giới hạn dưới
                a_lo = max(curr_best - _ASPIRATION_DELTA * 3, -INF)
            elif curr_best >= a_hi and a_hi < INF:
                # Fail-high: thực tế điểm cao hơn kỳ vọng => mở rộng giới hạn trên
                a_hi = min(curr_best + _ASPIRATION_DELTA * 3, INF)
            else:
                # Trong cửa sổ: kết quả tin cậy
                round_score = curr_best
                round_col = curr_col
                break

        if not round_aborted and round_score is not None:
            best_col = round_col
            prev_score = round_score
            prev_best_col = round_col

    return best_col


# %% [markdown]
# ## Kaggle Adapter

# %%
# =============================================================================
# Kaggle Adapter - chuyển đổi định dạng dữ liệu Kaggle sang bitboard
# =============================================================================


def _is_default_config(rows, cols, inarow):
    """Kiểm tra cấu hình chuẩn: 7 cột x 6 hàng, thắng khi 4 liên tiếp."""
    return rows == BH and cols == BW and inarow == 4


def _kaggle_to_bitboard(board, rows, cols, mark):
    """Chuyển bàn cờ 1D của Kaggle sang cặp bitboard.

    Kaggle dùng mảng 1D theo hàng ngang, từ trên xuống dưới:
      board[0..6]   = hàng trên cùng (hàng 5 trong bitboard)
      board[35..41] = hàng dưới cùng (hàng 0 trong bitboard)

    Trả về (my_board, opp_board, moves_made):
      my_board:   bitboard của người chơi đang đi (mark)
      opp_board:  bitboard của đối thủ
      moves_made: tổng số quân đã đặt trên bàn
    """
    my_board = opp_board = 0
    moves_made = 0

    for kaggle_row in range(rows):
        row_offset = kaggle_row * cols
        for col in range(cols):
            cell_value = board[row_offset + col]
            if cell_value == 0:
                continue
            # Kaggle đánh số hàng từ trên (0) xuống dưới (rows-1)
            # Bitboard đánh số từ dưới (0) lên trên, nên cần đảo ngược
            bitboard_row = rows - 1 - kaggle_row
            bit = 1 << (bitboard_row + col * BH1)

            if cell_value == mark:
                my_board |= bit
            else:
                opp_board |= bit
            moves_made += 1

    return my_board, opp_board, moves_made


# %% [markdown]
# ## WeakC4 Book

# %%
# WeakC4 book: precomputed perfect-play move table for Player 1.
#
# The book is generated by ``build_book.py`` from 2swap's WeakC4 graph. It has
# two halves:
#
#   - ``internal``: ``{(b0_red, b1_yellow): col}`` -- Red's forced move at
#     this exact Red-to-move position.
#   - ``leaf``:     ``{(b0_red, b1_yellow): packed_ss}`` -- a 21-byte 4-bit
#     packed 6x7 grid of "steady state" priority symbols. From the leaf or
#     any Red-to-move position in its subtree, querying the grid against the
#     current board picks Red's correct move.
#
# Priority codes (4 bits/cell):
#     0 = claimeven (blank, only valid on even ss-rows)
#     1 = red stone, 2 = yellow stone (never queried as a move target)
#     3 = claimodd ('|', only valid on odd ss-rows)
#     4 = urgent ('!')           5 = minus ('-')
#     6 = miai ('@', only if exactly one is playable)
#     7 = plus ('+')             8 = equal ('=')
#
# Priority order: urgent, miai, claimeven, claimodd, plus, equal, minus.
# Always preceded by an immediate-win check and an opponent-block check.
# =============================================================================

_BOOK_BLOB = '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'

_BOOK = None

# Cache of the leaf we are currently following. WeakC4's correctness depends
# on Red staying on the steady-state leaf they first entered: switching to a
# different leaf mid-game just because its stones happen to be a subset of
# the current position breaks the proof and can lose the game (this bit us
# in episode 75922681 -- we hit a direct leaf at ply 11, then ply 15 a
# *deeper* matching leaf existed and we switched to it; its ss diagram
# disagreed and the resulting position was no longer in any winning subtree).
# The cache stores ``(leaf_red, leaf_yellow, packed_ss)`` for the active
# leaf, plus the last observed ``mp`` so we can detect a fresh game and reset.
_ACTIVE_LEAF = None
_LAST_MP = -1


def _load_book():
    """Decode the embedded book on first use; tolerate any decode failure."""
    global _BOOK
    if _BOOK is not None:
        return _BOOK
    if not _BOOK_BLOB or _BOOK_BLOB.startswith("__BOOK_BLOB"):
        _BOOK = {"internal": {}, "leaf": {}}
        return _BOOK
    try:
        raw = zlib.decompress(base64.b64decode(_BOOK_BLOB))
        _BOOK = pickle.loads(raw)
    except Exception:
        _BOOK = {"internal": {}, "leaf": {}}
    return _BOOK


def _unpack_ss(packed):
    """Unpack 21 bytes of 4-bit codes into a 6x7 list. Row 0 is the top."""
    flat = []
    for byte in packed:
        flat.append((byte >> 4) & 0xF)
        flat.append(byte & 0xF)
    return [flat[r * BW:(r + 1) * BW] for r in range(BH)]


# Priority order (codes), excluding immediate win/block which run first.
_SS_PRIORITY_ORDER = (4, 6, 0, 3, 7, 8, 5)


def _query_steady_state(b0, b1, ss_grid):
    """Translate the JS ``querySteadyState`` to the bitboard layout.

    Returns the 0-indexed column to play, or -1 if no rule fires. The board
    arguments are the live position (not the leaf's frozen board); the grid
    is the leaf's annotation, which remains valid for any descendant.
    """
    occ = b0 | b1
    valid_now = (occ + BOTTOM_ROW) & VALID_CELLS

    # Immediate win for Red.
    p_threats = _find_threats(b0) & ~b1
    win_now = p_threats & valid_now
    if win_now:
        return _col_of_mask(win_now)

    # Block Yellow's win.
    o_threats = _find_threats(b1) & ~b0
    o_wins = o_threats & valid_now
    if o_wins:
        return _col_of_mask(o_wins & -o_wins)

    # Build the (col, ss_row) candidate list once.
    candidates = []
    for col in range(BW):
        m = _col_mask(col, occ)
        if m == 0:
            continue
        bit_pos = (m & -m).bit_length() - 1
        board_row = bit_pos - col * BH1   # 0 = bottom
        ss_row = (BH - 1) - board_row     # 0 = top, matching pack_protobuf
        candidates.append((col, ss_row))

    for prio in _SS_PRIORITY_ORDER:
        if prio == 6:  # miai: pick only if exactly one cell qualifies
            miai_cols = []
            for col, ss_row in candidates:
                if ss_grid[ss_row][col] == 6:
                    miai_cols.append(col)
                    if len(miai_cols) > 1:
                        break
            if len(miai_cols) == 1:
                return miai_cols[0]
            continue
        for col, ss_row in candidates:
            if ss_grid[ss_row][col] != prio:
                continue
            if prio == 0 and ss_row % 2 != 0:
                continue   # claimeven only fires on even ss-rows
            if prio == 3 and ss_row % 2 != 1:
                continue   # claimodd only fires on odd ss-rows
            return col
    return -1


def _bottom_bit_in_col(occ, col):
    """Lowest empty cell in ``col`` as a bitmask, or 0 if the column is full."""
    return (occ + BOTTOM_ROW) & (FIRST_COLUMN << (BH1 * col))


def _walk_forward_for_leaf(b0, b1):
    """Reconstruct the entered leaf by replaying from the empty board.

    The WeakC4 graph is a tree (rooted at the empty board) where Red's edges
    are deterministic (one per node) and Yellow's edges branch. The current
    position must lie on exactly one root-to-leaf path -- once we hit Red's
    first leaf along that path, we lock onto it.

    We walk forward, using the book to play Red's moves and inferring
    Yellow's actual moves from the difference between simulated and real
    yellow bitboards. Returns (leaf_red, leaf_yellow, packed_ss) for the
    first leaf on the path, or ``None`` if the path leaves the book before
    a leaf is found.
    """
    book = _load_book()
    cur_red = cur_yel = 0
    cur_mp = 0
    target_mp = bin(b0 | b1).count("1")

    while cur_mp < target_mp:
        if cur_mp % 2 == 0:
            # Red's turn -- consult the book.
            col = book["internal"].get((cur_red, cur_yel))
            if col is None:
                packed = book["leaf"].get((cur_red, cur_yel))
                if packed is None:
                    return None
                return (cur_red, cur_yel, packed)
            bit = _bottom_bit_in_col(cur_red | cur_yel, col)
            if bit == 0:
                return None
            cur_red |= bit
        else:
            # Yellow's turn -- the move that was actually played is the one
            # whose target cell shows a yellow stone in ``b1``. The "target
            # cell" in column c is the lowest cell of c that is empty in the
            # current simulation. There is exactly one column whose target
            # cell is occupied by Yellow in the real position.
            occ = cur_red | cur_yel
            chosen = 0
            for col in range(BW):
                bit = _bottom_bit_in_col(occ, col)
                if bit and (bit & b1) and not (bit & cur_yel):
                    chosen = bit
                    break
            if chosen == 0:
                return None
            cur_yel |= chosen
        cur_mp += 1

    # Reached the target without seeing a leaf along the way.
    return None


def _book_move(b0, b1, mp):
    """Return Red's column from the WeakC4 book, or -1 if not covered."""
    global _ACTIVE_LEAF, _LAST_MP

    book = _load_book()

    # Detect a new game: ``mp`` decreased, or we are at the very start.
    if mp <= 1 or mp < _LAST_MP:
        _ACTIVE_LEAF = None
    _LAST_MP = mp

    # Direct internal hit -- play the prescribed move. We are still in the
    # forced "memorize" trunk, so don't latch onto a leaf yet.
    col = book["internal"].get((b0, b1))
    if col is not None:
        return col

    # Direct leaf hit -- this *is* the leaf we should follow from now on.
    packed = book["leaf"].get((b0, b1))
    if packed is not None:
        _ACTIVE_LEAF = (b0, b1, packed)
        return _query_steady_state(b0, b1, _unpack_ss(packed))

    # Already locked onto a leaf and it still applies? Use it.
    if _ACTIVE_LEAF is not None:
        lb0, lb1, packed = _ACTIVE_LEAF
        if (lb0 & b0) == lb0 and (lb1 & b1) == lb1:
            return _query_steady_state(b0, b1, _unpack_ss(packed))
        # Otherwise the cached leaf no longer matches (different game, or we
        # somehow drifted off the strategy). Fall through to recovery.
        _ACTIVE_LEAF = None

    # No cached leaf -- reconstruct by walking forward from the empty board.
    leaf = _walk_forward_for_leaf(b0, b1)
    if leaf is None:
        return -1
    _ACTIVE_LEAF = leaf
    return _query_steady_state(b0, b1, _unpack_ss(leaf[2]))


# %% [markdown]
# ## Generic Fallback (Non-Default Configs)

# %%
# =============================================================================
# Fallback tổng quát cho các cấu hình bàn cờ khác chuẩn (không phải 7x6/K=4)
# Dùng minimax alpha-beta trên lưới 2D với heuristic đếm cửa sổ.
# Yếu hơn đường bitboard nhưng hoạt động với mọi kích thước bàn cờ.
# =============================================================================


def _fallback_agent(board, rows, cols, inarow, mark, deadline):
    # Chuyển mảng 1D sang lưới 2D để dễ xử lý
    grid = [list(board[r * cols:(r + 1) * cols]) for r in range(rows)]
    opp = 3 - mark  # mark=1 thì opp=2, mark=2 thì opp=1

    def col_height(c):
        """Trả về hàng trống thấp nhất của cột c (-1 nếu đầy)."""
        for r in range(rows - 1, -1, -1):
            if grid[r][c] == 0:
                return r
        return -1

    def is_win(r, c, who):
        """Kiểm tra xem quân vừa đặt tại (r, c) có tạo thắng không."""
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):  # dọc, ngang, 2 đường chéo
            count = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] == who:
                count += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] == who:
                count += 1
                rr -= dr
                cc -= dc
            if count >= inarow:
                return True
        return False

    def winning_move(c, who):
        """Kiểm tra xem đi vào cột c có thắng ngay không."""
        h = col_height(c)
        if h < 0:
            return False
        grid[h][c] = who
        won = is_win(h, c, who)
        grid[h][c] = 0  # hoàn tác
        return won

    # Sắp xếp cột hợp lệ: ưu tiên cột gần giữa nhất
    valid_ordered = sorted(
        (c for c in range(cols) if col_height(c) >= 0),
        key=lambda c: abs(c - cols // 2),
    )
    if not valid_ordered:
        return 0

    # Thắng ngay hoặc chặn thắng của đối thủ (không cần tìm kiếm)
    for c in valid_ordered:
        if winning_move(c, mark):
            return c
    for c in valid_ordered:
        if winning_move(c, opp):
            return c

    def score_window(window):
        """Đánh giá 1 cửa sổ inarow ô liên tiếp.

        Cửa sổ có quân của cả hai bên = 0 điểm (đã bị chặn, vô dụng).
        Cửa sổ gần thắng (còn thiếu 1 quân) = điểm rất cao (10x).
        """
        m = window.count(mark)
        o = window.count(opp)
        if m and o:
            return 0   # cửa sổ bị chặn
        if m:
            return m * m * (10 if m + 1 == inarow else 1)
        if o:
            return -o * o * (10 if o + 1 == inarow else 1)
        return 0

    def heuristic():
        """Tổng điểm heuristic bằng cách quét tất cả cửa sổ trên bàn cờ."""
        s = 0
        # Quét ngang
        for r in range(rows):
            row_data = grid[r]
            for c in range(cols - inarow + 1):
                s += score_window(row_data[c:c + inarow])
        # Quét dọc
        for c in range(cols):
            col_data = [grid[r][c] for r in range(rows)]
            for r in range(rows - inarow + 1):
                s += score_window(col_data[r:r + inarow])
        # Quét chéo xuong
        for r in range(rows - inarow + 1):
            for c in range(cols - inarow + 1):
                s += score_window([grid[r + k][c + k] for k in range(inarow)])
        # Quét chéo lên
        for r in range(inarow - 1, rows):
            for c in range(cols - inarow + 1):
                s += score_window([grid[r - k][c + k] for k in range(inarow)])
        # Bonus cột giữa
        center_col = cols // 2
        for r in range(rows):
            v = grid[r][center_col]
            if v == mark:
                s += 3
            elif v == opp:
                s -= 3
        return s

    INF_F = 1 << 30
    aborted_flag = [False]  # dùng list để có thể thay đổi trong hàm lồng

    def minimax(depth, alpha, beta, who):
        """Minimax negamax với cắt tỉa alpha-beta (dành cho bàn cờ tổng quát)."""
        if time.monotonic() >= deadline:
            aborted_flag[0] = True
            return 0
        if depth == 0:
            h = heuristic()
            return h if who == mark else -h
        moves = sorted(
            (c for c in range(cols) if col_height(c) >= 0),
            key=lambda c: abs(c - cols // 2),
        )
        if not moves:
            return 0
        best = -INF_F
        for c in moves:
            r = col_height(c)
            grid[r][c] = who
            if is_win(r, c, who):
                grid[r][c] = 0
                return INF_F - (rows * cols - depth)  # thắng sớm hơn = điểm cao hơn
            score = -minimax(depth - 1, -beta, -alpha, 3 - who)
            grid[r][c] = 0
            if aborted_flag[0]:
                return 0
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break  # beta cutoff
        return best

    # Iterative deepening: tăng dần độ sâu cho đến khi hết thời gian
    best_col = valid_ordered[0]
    for depth in range(2, rows * cols + 1):
        if time.monotonic() >= deadline:
            break
        aborted_flag[0] = False
        round_best_score = -INF_F
        round_best_col = best_col
        alpha = -INF_F

        for c in valid_ordered:
            r = col_height(c)
            grid[r][c] = mark
            if is_win(r, c, mark):
                grid[r][c] = 0
                return c  # thắng ngay ở độ sâu này
            score = -minimax(depth - 1, -INF_F, -alpha, opp)
            grid[r][c] = 0
            if aborted_flag[0]:
                break
            if score > round_best_score:
                round_best_score = score
                round_best_col = c
            if score > alpha:
                alpha = score

        if not aborted_flag[0]:
            best_col = round_best_col

    return best_col


# %% [markdown]
# ## Entry Point

# %%
# Entry point.
# =============================================================================


def _get(o, key, default=None):
    if hasattr(o, key):
        return getattr(o, key)
    try:
        return o[key]
    except (TypeError, KeyError):
        return default


def agent(observation, configuration):
    """Kaggle ConnectX entry point. Returns the column index to play."""
    start = time.monotonic()

    board = _get(observation, "board") or []
    mark = _get(observation, "mark", 1)
    rows = _get(configuration, "rows", 6)
    cols = _get(configuration, "columns", 7)
    inarow = _get(configuration, "inarow", 4)

    # Kaggle exposes per-action timeouts on newer builds; fall back to a
    # conservative budget when missing.
    budget = 0.85
    act_to = _get(configuration, "actTimeout") or _get(configuration, "timeout")
    if isinstance(act_to, (int, float)) and act_to > 0.5:
        budget = max(0.5, min(5.0, float(act_to) - 0.15))

    deadline = start + budget

    try:
        if _is_default_config(rows, cols, inarow):
            b0, b1, mp = _kaggle_to_bitboard(board, rows, cols, mark)
            if mp == 0:
                return cols // 2  # Hardcoded opening: drop in the centre.
            # Player-1 fast path: try the WeakC4 book. The book is built on
            # the assumption that Red plays centre on move 1; positions where
            # that didn't happen will simply miss and fall through to search.
            if mark == 1:
                book_col = _book_move(b0, b1, mp)
                if book_col >= 0 and _col_mask(book_col, b0 | b1):
                    return book_col
            return _solve_default(b0, b1, mp, deadline)
        return _fallback_agent(board, rows, cols, inarow, mark, deadline)
    except Exception:
        # Defensive: under no circumstance return an invalid column.
        for c in range(cols):
            if board[c] == 0:
                return c
        return 0


if __name__ == "__main__":
    # Tiny smoke test so the file can be sanity-checked locally.
    class _Obs:
        def __init__(self, board, mark):
            self.board = board
            self.mark = mark

    class _Cfg:
        rows = 6
        columns = 7
        inarow = 4

    empty = [0] * 42
    print("opening on empty board:", agent(_Obs(empty, 1), _Cfg()))

