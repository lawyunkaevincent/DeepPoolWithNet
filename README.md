# write report.json next to the net file (into the same folder):
python RLTesting/CleanPath/clean_path.py -n D:\6Sumo\FinalDQNwithPER\KLCC_Map\osm.net.xml -o D:\6Sumo\FinalDQNwithPER\KLCC_Map\reports.json

# or pass the folder and let the script name it connectivity_report.json:
python RLTesting/CleanPath/clean_path.py \
  -n "D:\6Sumo\2026-04-23-13-11-28\osm.net.xml" \
  -o "D:\6Sumo\2026-04-23-13-11-28"
