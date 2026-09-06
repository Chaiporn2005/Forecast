# L2 + Soft Monotonic — MTBLS242

This folder tests whether a **soft** monotonic constraint improves the current MTBLS242 responder forecast. It is a separate analysis from the original comparison folder and does not rewrite the earlier results.

## ผลหลัก

| โมเดล | Mean outer-fold PR-AUC | Conditional 95% interval |
|---|---:|---:|
| L2 | 0.686 | 0.586–0.813 |
| Monotonic + L2 | 0.684 | 0.585–0.809 |
| L2 + Soft Monotonic | **0.679** | 0.579–0.805 |

ผลจริงจาก 20 รอบคือ L2 + Soft Monotonic = **0.679**, Monotonic + L2 = **0.684** และ L2 = **0.686** Soft penalty ถูกเลือกมากกว่า 0 ใน 67/100 outer folds แต่ไม่ได้เพิ่มคะแนนเฉลี่ยในชุดข้อมูลนี้

การทดลองนี้ใช้ผู้ป่วย 71 คนและ target เดิมทุกประการ: responder คือผู้ที่มี metabolite อย่างน้อย 9 จาก 11 ตัวเปลี่ยนในทิศทางที่กำหนดหลัง 12 เดือน

## วิธีการ

โมเดล soft monotonic ใช้ objective:

`mean logistic loss + (lambda/2) * sum(w²) + (rho/2) * sum(max(0, -direction*w)²)`

`lambda` คือ L2 regularization และ `rho` คือความแรงของ soft monotonic penalty เมื่อ coefficient ฝ่าฝืนทิศทางที่กำหนด โมเดลไม่บังคับให้ทุก coefficient ทำตาม sign แบบ hard constraint จึงสามารถยอมให้ข้อมูลฝ่าฝืนได้ถ้าช่วยลด loss มากพอ

เลือกทั้ง `lambda` และ `rho` ใน inner folds เท่านั้น โดยใช้ grid: `lambda = 0.001, 0.01, 0.1, 1, 10` และ `rho = 0, 0.001, 0.01, 0.1, 1, 10` ค่า `rho = 0` เป็นกรณี L2 ที่ไม่มี soft penalty และถูกเก็บไว้เป็น sensitivity check

ใช้ repeated nested 5-fold CV จำนวน 20 รอบ (100 outer test folds) โดยใช้รายชื่อผู้ป่วยและ split เดียวกันสำหรับ Soft Monotonic + L2, Monotonic + L2 และ L2 ทุกโมเดล แต่ละคนมีคำทำนาย held-out หนึ่งครั้งต่อรอบต่อโมเดล ไม่มีการคัดตัวแปรจากทั้ง cohort และ scaler fit เฉพาะ training partition

ช่วงความเชื่อมั่นใช้ 2,000 paired subject-bootstrap draws ระดับผู้ป่วย ใช้ multiplicity เดียวกันทุกโมเดลและทุกรอบ ค่าช่วงจึงเป็น conditional interval ของคำทำนาย CV ที่บันทึกไว้ ไม่ใช่ coverage interval ที่พิสูจน์การ generalize ไปยังผู้ป่วยใหม่ และไม่มี external validation ในชุดนี้

## การแปลผล

หาก Soft Monotonic ได้ค่าเฉลี่ยสูงกว่าเล็กน้อย แต่ช่วง paired difference คร่อมศูนย์ จะยังสรุปว่า soft constraint ดีกว่าไม่ได้ หากค่า `rho` ถูกเลือกเป็นศูนย์บ่อย แสดงว่าการบังคับทิศทางไม่ได้ช่วยในข้อมูลชุดนี้ หาก `rho > 0` แต่คะแนนใกล้ L2 โมเดล soft constraint อาจเป็นทางเลือกที่สมดุลระหว่างการทำนายและความสอดคล้องทางชีววิทยา

ผลทั้งหมดต้องใช้เป็น proof of concept เพราะมีผู้ป่วยน้อยและ target เป็น metabolic surrogate ที่คำนวณจากการเปลี่ยนแปลงของ metabolite ซึ่งมี baseline เป็น input ด้วย ยังไม่ใช่หลักฐานประสิทธิภาพทางคลินิก ต้องมี external cohort, calibration, clinical endpoint และ prospective evaluation ก่อน

## ไฟล์

- `Results/summary.csv` — คะแนนเฉลี่ย ช่วงความเชื่อมั่น และผลต่างแบบจับคู่
- `Results/outer_fold_metrics.csv` — คะแนนของทุก outer fold
- `Results/heldout_predictions.csv` — คำทำนายของผู้ป่วยทุกคน ทุกโมเดล ทุก repeat
- `Results/inner_tuning_scores.csv` — คะแนน candidate ทุก inner fold
- `Results/inner_fold_membership.csv` — สมาชิก train/validation ตรวจ leakage
- `Results/subject_bootstrap_draws.csv` — bootstrap draws ที่ใช้สร้าง intervals
- `Results/protocol.json` — seeds, grids, hashes และนิยามช่วงความเชื่อมั่น
- `Results/run_checks.json`, `Results/verification.json` — ผลตรวจสอบ
- `Graph/soft_monotonic_confidence_intervals.png` — กราฟเปรียบเทียบพร้อม intervals
- `Graph/soft_monotonic_repeat_stability.png` — ความเสถียรระหว่าง repeats

## Run

```sh
python run_soft_monotonic.py
python verify_results.py
```

The script uses the fixed MTBLS242 data and implementation in the parent comparison folder. It creates no external validation result and does not fabricate missing external data.
