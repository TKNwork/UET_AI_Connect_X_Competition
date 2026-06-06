Hướng dẫn triển khai Dự án theo nhóm - Hạn nộp: Tuần 15
1. Mục tiêu

Xây dựng AI agent chơi ConnectX
Áp dụng các thuật toán AI đã được học (Minimax, Alpha-Beta pruning, heuristic…)
Triển khai agent trong môi trường thực tế
Đánh giá qua thi đấu trực tiếp giữa các nhóm, tính điểm trên leaderboard
2. Luật chơi ConnectX

- Bàn cờ: rows x columns (ví dụ 6x7)

- Mỗi lượt: thả quân vào một cột, quân sẽ rơi tới ô trống thấp nhất của cột đó

- Sau khi thả quân sẽ xảy ra một trong bốn khả năng:

Nếu cột đã đầy hoặc nằm ngoài phạm vi bàn cờ, agent lập tức thua
Nếu quân vừa đặt tạo thành 4-in-a-row (ngang/dọc/chéo), agent thắng
Nếu tất cả ô đều kín mà không ai thắng, ván đấu hòa
Nếu không có sự kiện nào ở trên xảy ra, lượt đi chuyển cho đối thủ
Kết thúc:

- Thắng: có 4 quân liên tiếp

- Thua: đối thủ đạt 4 quân liên tiếp

- Hòa: bàn đầy

3. Mô hình thi đấu trên Kaggle

- Mỗi nhóm upload 1 file Python (agent.py)

- Kaggle tự động ghép trận và tính điểm

Điểm leaderboard:

- Dạng rating (không phải số trận thắng)

- Ban đầu ~600, thay đổi theo kết quả trận đấu

- Điểm sẽ thay đổi và cập nhật liên tục theo thời gian

4. Cách submit lên Kaggle

Bước 1: Tạo tài khoản Kaggle

Truy cập: https://www.kaggle.comLinks to an external site.
Đăng ký tài khoản (có thể dùng Google/GitHub)
Bước 2: Tham gia competition

Truy cập: https://www.kaggle.com/competitions/connectxLinks to an external site.
Bấm “Join Competition”
Đồng ý các điều khoản
Đổi tên nhóm thành UET_INT3401E6_[số thứ tự nhóm]
Bước 3: Chuẩn bị file

Mỗi nhóm chuẩn bị 1 file submission.zip, bao gồm “agent.py” và “model.pth” (nếu có)
Bên trong file agent.py cần chứa hàm:
         def agent(observation, configuration):

                return action

Ví dụ agent đơn giản

        import random

        def agent(observation, configuration):

               valid_moves = [c for c in range(configuration.columns) if observation.board[c] == 0]

               return random.choice(valid_moves)

 

5. Yêu cầu bài tập

Mỗi nhóm gồm tối đa 03 sinh viên phát triển một agent riêng:
Đăng ký nhóm tại ĐÂYLinks to an external site.
Sử dụng email VNU đăng ký
Áp dụng các thuật toán AI đã được học (Minimax, Alpha-Beta pruning, heuristic…)
Output:
Mã nguồn tệp agent.py
Báo cáo (ở dạng ppt slide) trình bày về ý tưởng thuật toán, phương pháp cải tiến thuật toán gốc
6. Tiêu chí chấm điểm

Điểm Leaderboard trên Kaggle: 30%: 
Ý tưởng thuật toán: 40%
Trình bày báo cáo, mã nguồn: 30%
Hạn nộp báo cáo: Tuần 15
03-05 nhóm sẽ được chọn báo cáo trước lớp vào tuần 16
