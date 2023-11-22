# Run -h option to see the meaning of all arguments
# python run_amber_RAMD.py -h

python run_amber_RAMD.py ../data/hsp90_INH.pdb ../data/hsp90_INH.prmtop \
    "569,583,605,617,1266,1292,1299,1374,1440,1459,1499,1849,1872,1892,2256,2295,2352,2557" \
    --ligand_resname=INH --cuda_device_index=0 --temperature=298.15 \
    --target_pressure=1.0 --ramd_log_filename=hsp90_ramd.log
    

# Shorthand
#python run_amber_RAMD.py ../data/hsp90_INH.pdb ../data/hsp90_INH.prmtop \
#    "569,583,605,617,1266,1292,1299,1374,1440,1459,1499,1849,1872,1892,2256,2295,2352,2557" \
#    -L INH -c 0 -T 298.15 -P 1.0 -R hsp90_ramd.log
