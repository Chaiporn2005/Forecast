# ผลลัพธ์ Forecast Model: Monotonic Constraints + PR-AUC

## โจทย์

ใช้ข้อมูล serum NMR ก่อนผ่าตัด เพื่อพยากรณ์ว่าในอีก 12 เดือน ผู้ป่วยจะเป็น
**strong metabolic responder** หรือไม่ โดยนิยาม responder ว่าเมตาโบไลต์อย่างน้อย
9 จาก 11 ตัว (≥80%) เปลี่ยนไปในทิศทางที่ดีขึ้นตามหลักฐานด้าน obesity
metabolomics

## วิธีประเมิน

- ผู้ป่วยที่มีข้อมูลทั้ง preop และ 12 เดือน: **71 คน**
- Strong responders: **31 คน (43.7%)**
- โมเดล: monotonic logistic regression
- Constraints: **เพิ่ม 8 / ลด 3 / ไม่กำหนด 10**
- Nested stratified **5-fold cross-validation**
- ประเมินเฉพาะผู้ป่วยใน held-out fold
- Metric หลัก: **PR-AUC (Average Precision)**

## ผลลัพธ์

| Metric | Result |
|---|---:|
| Held-out OOF PR-AUC | **0.671** |
| Mean fold PR-AUC | **0.653 ± 0.184** |
| No-skill baseline (prevalence) | **0.437** |
| Monotonicity violations | **0** |

## การแปลผล

โมเดลมี PR-AUC สูงกว่า prevalence baseline ประมาณ **1.54 เท่า** แสดงว่า NMR
ก่อนผ่าตัดมีสัญญาณบางส่วนในการจัดลำดับผู้ที่มีแนวโน้มตอบสนองทางเมตาบอลิซึมสูง
อย่างไรก็ตาม ผลระหว่าง fold แปรปรวนมาก เพราะมีผู้ป่วยเพียง 71 คน จึงควรถือเป็น
**proof of concept** และต้องตรวจสอบกับ cohort ภายนอกก่อนนำไปใช้จริง

ผลนี้เป็นการพยากรณ์การเปลี่ยนแปลงของ metabolite profile หลังผ่าตัด ไม่ใช่การ
พยากรณ์น้ำหนักลด การหายจากโรคอ้วน หรือเหตุการณ์ทางคลินิกโดยตรง
