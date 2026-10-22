# Run -h option to see the meaning of all arguments
# python run_amber_RAMD.py -h

python run_amber_RAMD.py ../data/hsp90_INH.pdb ../data/hsp90_INH.prmtop \
    --receptor_indices="569,583,605,617,1266,1292,1299,1374,1440,1459,1499,1849,1872,1892,2256,2295,2352,2557" \
    --ligand_indices="3259,3260,3261,3262,3263,3264,3265,3266,3267,3268,3269,3270,3271,3272,3273,3274,3275,3276,3277,3278,3279,3280,3281,3282,3283,3284,3285,3286,3287,3288" \
    --cuda_device_index=0 --temperature=298.15 \
    --target_pressure=1.0 --ramd_log_filename=hsp90_ramd.log


# Shorthand
#python run_amber_RAMD.py ../data/hsp90_INH.pdb ../data/hsp90_INH.prmtop \
#    "569,583,605,617,1266,1292,1299,1374,1440,1459,1499,1849,1872,1892,2256,2295,2352,2557" \
#    -L INH -c 0 -T 298.15 -P 1.0 -R hsp90_ramd.log
