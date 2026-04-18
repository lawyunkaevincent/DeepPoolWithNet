Step 1 — Collect the dataset (run heuristic in SUMO)
This runs the simulation with the heuristic policy and records decisions to a CSV.
cd d:/FYP/DQNLargeScale1/DQNImitation/ImitationDL

python collect_imitation_dataset.py --cfg ../../SunwaySmallMap/osm.sumocfg --step-length 1.0 --dataset-out imitation_dataset.csv
Add --gui if you want the SUMO window to appear.

Skip Step 1 if imitation_dataset.csv already exists and is up to date.

Step 2 — Train the model

python train_imitation_model.py --dataset imitation_dataset.csv --output-dir artifacts/imitation_model --epochs 60

The trained model is saved to DQNImitation/ImitationDL/artifacts/imitation_model/ — you already have a previous run there (imitation_model.pt).

Optional train args worth knowing:
Arg	Default	Notes
--epochs	60	Increase if underfitting
--patience	10	Early stopping
--hidden-dims	256,128	e.g. --hidden-dims 512,256,128
--device	auto (cuda/cpu)	
--batch-size	64	

After training, run the imitation model against SUMO with:
python run_imitation_policy.py --cfg ../../SunwaySmallMap/osm.sumocfg --model-dir artifacts/imitation_model --step-length 1