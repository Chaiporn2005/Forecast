# L2_Predict — L2 + clinical predictors + NMR

โฟลเดอร์นี้เตรียม pipeline สำหรับเพิ่มข้อมูลคลินิกจริงเข้ากับ NMR ของ MTBLS242
โดยใช้ **L2-regularized logistic regression** เป็นโมเดลหลัก และเปรียบเทียบกับ
**L2 (NMR only)** และ **Monotonic + L2 (NMR only)** บนผู้ป่วยชุดเดียวกัน

## สถานะปัจจุบัน

ในไฟล์ MTBLS242 ที่อยู่ใน repository มีข้อมูล sample ID, ชนิดตัวอย่าง และ time point
เท่านั้น ไม่มีอายุ เพศ BMI ชนิดการผ่าตัด เบาหวาน ยา ความดัน ไขมัน น้ำตาล หรือผลเลือด
รายบุคคล ดังนั้นยังไม่มีผล `L2 + clinical + NMR` หรือ external validation ที่รายงานได้
และ pipeline จะหยุดทำงานหากไม่มีไฟล์ clinical จริง เพื่อป้องกันการสร้างผลลวง

ข้อมูลต้นทางอย่างเป็นทางการอยู่ที่ [MetaboLights MTBLS242 FTP](https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS242/)
ซึ่งระบุ study factor เป็น time point และไม่ได้ให้ clinical table รายบุคคลในไฟล์ sample/assay

## ตัวแปรที่ต้องเตรียม

สร้างไฟล์ CSV หนึ่งแถวต่อผู้ป่วย โดยใช้ `subject_id` ให้ตรงกับรหัส 4 หลักใน MTBLS242
(เช่น `0003`) และมีคอลัมน์ต่อไปนี้:

| กลุ่ม | คอลัมน์ | หน่วย/รูปแบบที่แนะนำ |
|---|---|---|
| พื้นฐาน | `age` | ปี, ตัวเลข |
| พื้นฐาน | `sex` | หมวดหมู่ เช่น `female`, `male` |
| พื้นฐาน | `bmi` | kg/m² |
| การรักษา | `surgery_type` | เช่น `sleeve`, `proximal_RYGB`, `distal_RYGB` |
| โรค/ยา | `diabetes` | 0/1 หรือ `yes`/`no` |
| โรค/ยา | `medication_any` | 0/1 หรือ `yes`/`no`; ระบุยาที่ใช้เพิ่มเติมได้ในไฟล์เสริม |
| สัญญาณชีพ | `systolic_bp`, `diastolic_bp` | mmHg |
| ไขมัน | `total_cholesterol`, `hdl_cholesterol`, `triglycerides` | ใช้หน่วยเดียวกันทั้งชุด (ระบุใน metadata) |
| น้ำตาล | `glucose` | ใช้หน่วยเดียวกันทั้งชุด (ระบุใน metadata) |
| ผลเลือดก่อนผ่าตัด | `creatinine`, `bun`, `uric_acid`, `albumin` | ใช้หน่วยเดียวกันทั้งชุด |

สามารถมี missing values ได้ แต่ต้องรายงานจำนวน missing และห้ามกรอกค่าประมาณจากข้อมูล
ภายนอกโดยไม่บันทึกแหล่งที่มา โปรแกรมจะ fit median/mode imputation และ scaler
**ภายใน training partition ของแต่ละ inner fold เท่านั้น**

เริ่มจาก header ใน [`clinical_input_template.csv`](clinical_input_template.csv)
แล้วเติมข้อมูลผู้ป่วยจริง หรือสร้างแถวตาม subject IDs ด้วย:

```sh
python validate_setup.py --write-template
```

## การประเมิน

```sh
# ตรวจสอบว่าข้อมูลพร้อมหรือยัง (ไม่ fit โมเดล)
python validate_setup.py --clinical clinical_mtbls242.csv

# รัน repeated nested CV: 20 repeats × outer 5 folds × inner 5 folds
python run_clinical_l2.py --clinical clinical_mtbls242.csv

# เมื่อมี independent cohort จริง ให้ใช้ไฟล์ที่มี target, NMR 21 ตัว และ clinical columns
python run_clinical_l2.py --clinical clinical_mtbls242.csv --external external_mtbls242.csv
```

การแบ่งผู้ป่วยใช้ seeds และฟังก์ชัน split เดียวกับผลเดิมใน
`Monotonic vs L2 vs Elastic Net vs SVM` และใช้ target เดิมทุกประการ:
responder = metabolite อย่างน้อย 9 จาก 11 ตัวเปลี่ยนในทิศทางที่กำหนดหลัง 12 เดือน

- tuning lambda อยู่ใน inner folds เท่านั้น
- clinical preprocessing อยู่ใน fit ของแต่ละ inner/outer fold เท่านั้น
- คะแนนหลักคือ PR-AUC (Average Precision) ของ outer held-out folds
- confidence interval เป็น paired subject-bootstrap interval ของ held-out predictions
- external validation ต้องเป็น cohort อิสระ ห้ามใช้ผู้ป่วย MTBLS242 ซ้ำ

ไฟล์ผลลัพธ์จะถูกเขียนใน `Results/` และกราฟ PNG ใน `Graph/` เมื่อมี clinical data จริง
ไฟล์ `Results/availability_report.json` บันทึกเหตุผลที่ยังไม่สามารถสร้างผลตอนนี้ได้

## ผลอ้างอิงจาก NMR-only ที่มีอยู่แล้ว

ผล repeated nested CV เดิม (ผู้ป่วย 71 คน, 20 repeats) ใช้เป็น comparator เท่านั้น:

| โมเดล | Mean outer-fold PR-AUC |
|---|---:|
| L2 (NMR only) | 0.686 |
| Monotonic + L2 (NMR only) | 0.684 |

ค่าดังกล่าวไม่ได้แปลว่า clinical model ดีกว่าหรือแย่กว่า จนกว่าจะเติม clinical table
และรัน pipeline นี้สำเร็จ
