# Navigating DMES Uncertainty with coevolution

## Run

1. Clone the repo
2. Install dependencies in your environment (conda or otherwise)

```sh
pip install -r requirements.txt
```

3. Set up case studies under `test_cases/<yourprojectname>/inputs/` following existing ones as templates and referring to the schema at `schema/schema.toml`

4. To run the modified POET, look at `main.py` and uncomment relevant section before running

5. To run the visualiser, run `app.py` which is setup to run the boiler_chp example. This can be changed under visualiser/setup.py. One also needs to specify parameters of both encodings in that file. Once running, the visualiser will be loaded at `http://127.0.0.1:8050/`


## Credits

Much of this code was created by or in collaboration with Jack Hawthorne as part of his Master Thesis.