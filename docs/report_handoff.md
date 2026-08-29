# Handoff hoàn thiện báo cáo 35-45 trang

## Trạng thái

Báo cáo Word/PDF nguồn không được tự động ghi đè trong gate code. Chỉ bắt đầu điền kết quả khi `report_facts.json` và bộ report assets đã được final evaluation sinh thật.

Locked split SDNET2018 đã có và đã pass integrity, nhưng chưa có run E1-E5 chính thức hoặc `report_facts.json`; trạng thái vẫn là **CHƯA SẴN SÀNG NỘP**. Không được suy metric mô hình từ số liệu dữ liệu hoặc smoke.

## Bố cục đề xuất

1. **Giới thiệu:** bối cảnh, vấn đề, mục tiêu Accuracy >=0,92, câu hỏi nghiên cứu, phạm vi và disclaimer.
2. **Cơ sở lý thuyết/tổng quan:** CNN, transfer learning, MobileNetV2, preprocessing, augmentation, metric, Grad-CAM, nghiên cứu liên quan.
3. **Phân tích và phương pháp:** yêu cầu, I/O, kiến trúc, SDNET2018, source-group split, E1-E5, test-lock.
4. **Xây dựng và thực nghiệm:** môi trường thật, audit dữ liệu, huấn luyện, lựa chọn, kết quả test, latency, Grad-CAM, error analysis, ảnh tự chụp.
5. **Hệ thống/kết luận:** Streamlit, kiểm thử, mức đạt mục tiêu, hạn chế, hướng phát triển.

Theo tài liệu đề tài, báo cáo cuối cần 35-45 trang. Không kéo dài bằng output log/code thô; đưa chi tiết config, manifest schema và log dài vào phụ lục.

## Dữ liệu cần lấy tự động

| Nội dung | Nguồn |
|---|---|
| Số lượng/phân bố/file lỗi/trùng | audit gốc + conflict report + dataset summary sau curation |
| Split và hash | manifests + split audit |
| Môi trường | `environment.json`, pip freeze |
| E1-E5 | comparison table + facts |
| Metric test/FP/FN | final metrics/predictions |
| Curves/confusion matrix | report figures |
| Threshold | selection lock + threshold curve |
| Latency | benchmark table/raw timing |
| Grad-CAM | TP/TN/FP/FN grid |
| Ảnh thực tế | real-image report |

## Quy tắc viết kết luận

- Viết “đạt Accuracy >=0,92” chỉ khi facts/test artifact chứng minh và ghi threshold.
- Nếu không đạt, nêu số thật và nguyên nhân có bằng chứng; không thay test/split.
- Không gọi Grad-CAM là segmentation hoặc định vị pixel.
- Phân biệt rõ kết quả test chuẩn với ảnh thực tế.
- Không so trực tiếp metric với bài báo dùng dataset/split/nhiệm vụ khác mà không cảnh báo.

## Việc còn do nhóm thực hiện

- Điền đúng tên 4-5 sinh viên, MSSV, lớp, khóa, GVHD và biểu mẫu trường.
- Tạo Git commit sạch, sau đó chạy Gate D trên GPU/runtime thật với locked `split_v1`.
- Thu thập/gán nhãn ảnh thực tế có quyền sử dụng.
- Viết thảo luận dựa trên error cases thật.
- Render/soát trang, trích dẫn, chính tả và xác nhận 35-45 trang.
- Xin GVHD duyệt nội dung và hình thức trước nộp.

## Checklist đối chiếu cuối

- [ ] Tên đề tài Việt/Anh nhất quán.
- [ ] Mọi metric khớp `report_facts.json`.
- [ ] Hash/run ID ghi ở phụ lục hoặc bảng provenance.
- [ ] Hình/bảng đánh số, caption và nguồn.
- [ ] SDNET2018 và MobileNetV2/Grad-CAM được trích dẫn.
- [ ] Có Accuracy, Precision, Recall, F1, confusion matrix, FP/FN.
- [ ] Có train/validation curves và latency.
- [ ] Có TP/TN/FP/FN Grad-CAM và ảnh thực tế riêng.
- [ ] Có disclaimer và giới hạn.
- [ ] PDF render sạch, 35-45 trang, không placeholder/TODO.
