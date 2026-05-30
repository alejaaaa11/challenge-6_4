#!/bin/bash
# Download CCD data files from NCES — Challenge 6, Group 4
# Same files as Challenge 5

cd /workspaces/challenge-6_4/data

echo "Downloading CCD data files from NCES..."

wget -q "https://nces.ed.gov/ccd/data/zip/ccd_sch_029_2223_w_1a_083023.zip" -O tmp.zip && unzip -o tmp.zip && rm tmp.zip && echo "✓ Directory"

wget -q "https://nces.ed.gov/ccd/data/zip/ccd_sch_033_2223_l_1a_083023.zip" -O tmp.zip && unzip -o tmp.zip && rm tmp.zip && echo "✓ Lunch Eligibility"

wget -q "https://nces.ed.gov/ccd/data/zip/ccd_sch_059_2223_l_1a_083023.zip" -O tmp.zip && unzip -o tmp.zip && rm tmp.zip && echo "✓ Staff"

wget -q "https://nces.ed.gov/ccd/data/zip/ccd_sch_129_2223_w_1a_083023.zip" -O tmp.zip && unzip -o tmp.zip && rm tmp.zip && echo "✓ School Characteristics"

wget -q "https://nces.ed.gov/ccd/data/zip/ccd_sch_052_2223_l_1a_083023.zip" -O tmp.zip && unzip -o tmp.zip && rm tmp.zip && echo "✓ Membership (large file, please wait...)"

echo ""
echo "Done. All 5 files ready in data/"
ls -lh *.csv
