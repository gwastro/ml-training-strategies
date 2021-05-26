python ./pycbc_create_ml_test_data.py \
--config-file <full-path-to-wf.ini> \
--create-injections \
--cumsum-injection-times \
--workers 720 \
--injection-config <full-path-to-injection.ini>
#720 workers = each worker processes 1 hour of data

#Use the following command to submit the job:
#pycbc_submit_dax --no-grid --no-create-proxy --dax test_data_generation.dax --enable-shared-filesystem
