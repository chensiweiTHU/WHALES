FFNet_workdir=$2 # '/home/yuhaibao/FFNet-VIC3D'
export PYTHONPATH=$PYTHONPATH:${FFNet_workdir}

DELAY_K=$3
DATA=${FFNet_workdir}'/data/DAIR-V2X-C/cooperative-vehicle-infrastructure'
VAL_DATA_PATH=${FFNet_workdir}'/data_process/dair-v2x/flow_data_jsons/flow_data_info_val_'${DELAY_K}'.json'
OUTPUT="../cache/vic-feature-flow"
VEHICLE_MODEL_PATH=${FFNet_workdir}'/work_dirs/config_basemodel/latest.pth'
VEHICLE_CONFIG_NAME=${FFNet_workdir}'/configs_CooperativePerception/ffnet/config_basemodel.py'

CUDA_VISIBLE_DEVICES=$1 

python eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model feature_flow \
  --test-mode $4 \
  --dataset vic-sync \
  --val-data-path $VAL_DATA_PATH \
  --veh-config-path $VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH \
  --device $CUDA_VISIBLE_DEVICES \
  --pred-class car \
  --sensortype lidar \
  --extended-range 0 -39.68 -3 100 39.68 1

  ## bash scripts/lidar_feature_flow.sh 0 ~/Projects/v2x-agents 0 'OriginFeat'