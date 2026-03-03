# Model Card - Income Classification Based on Census data

## Model Details
- Model type: RandomForestClassifier(random_state=42) (scikit-learn).
- Task: Binary classification to predict whether salary is >50K or <=50K (Adult Census Income–style target).
The model is served via a FastAPI app for online inference.

## Intended Use
Intended for educational/demo purposes: showcasing small ML pipeline + API deployment (Render) + CI/CD (GitHub)
- Expected use: batch/offline evaluation and simple online predictions through the API.
Not intended for high-stakes decisions (e.g., employment, credit, housing) without extensive additional validation and 
governance.

## Training Data
- Dataset: census.csv, derived from the 1994 US Census data (Adult Census Income dataset).
- Description: Records include demographic and employment-related features (e.g., education/work-related attributes), 
used to predict whether an individual’s income exceeds $50K/year. 
- Label: salary (binary outcome; typically >50K vs <=50K)
- Split: 80/20 train/test split (80% training, 20% evaluation)
- Preprocessing: Categorical features are encoded and numerical features are passed through your processing pipeline 
before training/inference .

## Evaluation Data
- Evaluation dataset: The 20% held-out split from census.csv created during your 80/20 split
- Additional evaluation: Slice metrics computed across the education feature (16 category slices), 
including per-slice counts and precision/recall/F-beta

## Metrics
Overall performance (no slicing):

- Precision: 0.7382239382239382
- Recall: 0.6256544502617801
- FBeta: 0.6772936592277718

Slice performance (feature: education, total slices: 16):
- Strong slices (high F-beta): Prof-school (F=0.8941), Doctorate (F=0.8714), Masters (F=0.8285)
- Weaker slices: 1st-4th (F=0.0000; recall 0.0000), Preschool (F=0.0000; count=11), 12th (F=0.2500)
See [Slice Result](slice_output.txt)

Reliability note: Very small slices can produce unstable metrics; interpret with the provided per-slice count 
(e.g., Preschool count=11). 

## Ethical Considerations
- Bias & fairness: Performance varies across education slices, indicating potentially different error rates across subgroups and the need for fairness assessment before any real use.
- Data limitations: Census/income datasets can encode historical and societal biases; models trained on them may reproduce those patterns.
- Privacy: The dataset represents sensitive socioeconomic attributes; treat inputs/outputs as sensitive in logs and monitoring

## Caveats and Recommendations
- Outdated data: Trained on 1994 US Census-derived data, so feature→income relationships may not generalizable to 
today’s population or non‑US contexts
- Missing values: Census data has missing values as ?; even though preprocessing handles these consistently 
(treats as “Unknown”) but it can shift results
- Subgroup variability: education slice metrics vary a lot and some slices are tiny (e.g., Preschool count=11), 
so evaluating more slices (e.g., race, sex, native-country) would make evaluation strategy more reliable
