MEDCASE_SYSTEM_PROMPT = """You are a medical reasoning assistant.

Follow these rules STRICTLY:
1. First, reason internally and output it ONLY inside <think>...</think>.
2. Then output ONLY the final answer inside <answer>...</answer>.
3. Do NOT include the final answer inside the reasoning.
4. Base reasoning only on provided context, images, and established medical facts.
5. Explicitly state uncertainty if evidence is insufficient.
6. Use concise, professional clinical language.
7. Output plain text only. No markdown.

You MUST follow the output template exactly.
"""

FEW_SHOT_EXAMPLES = """\
CASE PRESENTATION:
A 47-year-old man presents to the emergency department with a headache. He suddenly developed a throbbing, bitemporal headache about 5 hours ago "out of nowhere." He has a history of migraine headaches, but he feels that this headache is significantly more painful than his typical migraines. The patient took his prescribed sumatriptan with no relief of his symptoms. He also endorses nausea and he reports that he vomited once before arrival in the emergency department. The patient denies any recent trauma to the head. His medical history is significant for migraines and hypertension. He has a 20-pack-year smoking history and a history of cocaine use. He drinks 5-6 beers per week. His temperature is 98.6°F (37°C), blood pressure is 147/91 mmHg, pulse is 62/min, and respirations are 12/min. On physical exam, he appears to be in moderate distress and has pain with neck flexion. He has no focal neurologic deficits. A head CT is performed and can be seen in Figure A. Based on the clinical presentation and imaging findings, in which space is this patient experiencing bleeding?

<think>
This patient presents with a sudden, severe headache and a head CT showing bleeding, which is consistent with a diagnosis of subarachnoid hemorrhage. Subarachnoid hemorrhages typically present with a sudden, severe headache referred to as a “thunderclap” headache, often described as the "worst headache of their life." Patients may also exhibit symptoms of meningeal irritation, such as nausea, vomiting, and nuchal rigidity. Although most cases are secondary to trauma, risk factors like hypertension and cocaine use can lead to non-traumatic presentations.

The head CT in Figure A demonstrates hyperintensity in the basal cisterns, indicating bleeding within the subarachnoid space. This space is located between the arachnoid mater and the pia mater. Therefore, the bleeding in this patient is occurring in the space between these two brain layers.
</think>
<answer>
Arachnoid mater and pia mater
</answer>

---

CASE PRESENTATION:
A 27-year-old woman presents for her first prenatal visit at an estimated gestational age of 16 weeks and 4 days by last menstrual period. She is presenting late to prenatal care due to significant anxiety about her pregnancy, as she and her husband had struggled with infertility for several years prior to conceiving. She reports experiencing nausea and vomiting for about 2 months, tender and swollen breasts, and cravings for foods she typically does not eat. She took a pregnancy test about 10 weeks ago and was too nervous to read the result immediately, and after 24 hours, she saw that it was positive. The patient has a past medical history of bulimia nervosa but has not been symptomatic in 2 years, and also had an appendectomy at age 15 for appendicitis. She is a Ph.D. student and her husband is a physician. Her vital signs are: temperature 98.5°F (36.9°C), pulse 75/min, blood pressure 122/76 mmHg, and respirations 13/min. The patient’s affect is normal and she is pleasant and excited about her pregnancy. Physical exam is notable for abdominal distension but is otherwise unremarkable. Ultrasound findings are shown in Figure A, and urine human chorionic gonadotropin is negative in the office. What is the most likely diagnosis?

<think>
The patient presents with signs and symptoms of pregnancy but a nongravid uterus on ultrasound and a negative pregnancy test. Therefore, the most likely diagnosis is pseudocyesis.

Pseudocyesis, or false pregnancy, most commonly presents with abdominal distension, breast tenderness, nausea, and other typical symptoms of pregnancy. Patients truly believe they are pregnant and may report positive pregnancy tests at home (perhaps due to false positives from waiting too long before reading the result) but office testing is negative. Ultrasound will be normal and show the absence of a developing fetus. The mainstay of treatment is explaining the diagnosis in a therapeutic manner and offering counseling to help patients cope.

Small reviews the evidence regarding the diagnosis and treatment of pseudocyesis, discussing how this disease is a heterogenous entity involving neuroendocrine changes and recommending clearly defining the cause of this disease in patients.

Figure A shows a transabdominal ultrasound with an empty cavity without signs of pregnancy (red circle). These findings are consistent with pseudocyesis.
</think>
<answer>
Pseudocyesis
</answer>

---
"""
