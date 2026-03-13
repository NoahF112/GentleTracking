# Sim2Sim instruction
- Create `conda` environment and install dependencies:
    ```bash
    conda env create -f environment.yaml
    conda activate test_sim2real
    ```

- Run `python src/simple_deploy_framework.py --help` to see instructions, by default, run following command will execute a `y1_v2` model dancing `dance1_subject1`

# Sim2Real instruction
- Methods needed to re-write are 
    - `_collect_state_variables`
    - `run`

- Start up process should follow `simple_deploy_framework.py`
