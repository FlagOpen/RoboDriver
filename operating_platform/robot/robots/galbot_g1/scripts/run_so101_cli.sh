conda activate op

python operating_platform/core/main.py \
    --robot.type=so101 \
    --record.repo_id="test_so101_baai" \
    --record.single_task="test my so101 arm."

