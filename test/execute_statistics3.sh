
for j in {1..99}
    cd ..
    cd pynever
    cd test
    j = `cat precision.txt`
    python3 update_precision.py
    cd ..
    cd ..
    cd NeVer_Jia_Riinard_quick
    for i in {1..99}
    do
        cd data
        python3 generate_x_seed_data.py $i
        cd ..
        python3 NeVer_attack.py $i
        python3 x0_generator.py $i
        python3 integer_linear_programming2.py $i $j
    done
