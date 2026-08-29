# Handoff slide bảo vệ

## Mục tiêu

Slide chỉ tóm tắt bằng chứng thật, không thay báo cáo và không dùng metric dự kiến. Dùng cùng `report_facts.json` với báo cáo.

Hiện chưa có metric E1-E5 chính thức hoặc `report_facts.json`, vì vậy slide kết quả cuối vẫn **CHƯA CÓ BẰNG CHỨNG** và hồ sơ **CHƯA SẴN SÀNG NỘP**.

## Dàn ý 10-12 slide

1. Trang bìa: đề tài, nhóm, GVHD.
2. Bối cảnh, bài toán và phạm vi an toàn.
3. Mục tiêu/câu hỏi nghiên cứu.
4. SDNET2018, phân bố và chống leakage theo source group.
5. Kiến trúc MobileNetV2 và pipeline chung.
6. Protocol E1-E5 và test-lock.
7. Learning curves/so sánh E1-E4.
8. Final metrics + confusion matrix + threshold E5.
9. Grad-CAM TP/TN/FP/FN.
10. Ảnh tự chụp và domain shift.
11. Demo/latency/kiểm thử.
12. Kết luận, hạn chế, hướng phát triển.

## Quy tắc trình bày

- Mỗi slide một thông điệp, chữ đủ lớn, không chụp màn hình bảng dày.
- Ghi split, threshold, cỡ mẫu và run/hash ngắn ở chú thích khi có metric.
- Dùng confusion matrix/ảnh Grad-CAM thật, không ảnh minh họa giả.
- Nếu Accuracy dưới 0,92, trình bày trung thực cùng phân tích.
- Không gọi heatmap là vùng nứt chính xác.
- Cảnh báo “khảo sát sơ bộ, không thay kỹ sư” xuất hiện ở slide kết quả/demo.

## Việc còn lại

- [ ] Thay thông tin nhóm/GVHD chính xác.
- [ ] Xuất hình từ `artifacts/report/` ở độ phân giải phù hợp.
- [ ] Đồng bộ mọi số với facts/report.
- [ ] Chèn QR/repository chỉ khi URL thật và quyền truy cập phù hợp.
- [ ] Diễn tập 5-7 phút theo `demo_script.md`.
- [ ] Có bản PDF và bản PPTX mở được offline.
