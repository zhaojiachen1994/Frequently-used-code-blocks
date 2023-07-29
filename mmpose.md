<details>
    <summary><strong>   test a customized dataset with evaluate  </strong></summary>
    
```python
    from mmpose.datasets import build_dataloader, build_dataset
    from mmpose.datasets import DATASETS
    import mmcv
    config_file = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/mouse/dataset_wholemouse.py"
    config = mmcv.Config.fromfile(config_file)
    dataset = build_dataset(config.data.train)
    #check the db
    ic(len(dataset.db))
    ic(dataset.db[0].keys())
    a = dataset.__getitem__(4)
    ic(a.keys())

    # check dataset evaluate
    results = convert_db_to_output(dataset.db)
    infos = dataset.evaluate(results, metric='mAP')

    # check the dataloader
    dataloader = build_dataloader(dataset, samples_per_gpu=2, workers_per_gpu=2)
    _, a = next(enumerate(dataloader))
    ic(a.keys())

  

```

</details>

------------------------------------------------------------------------------------------------------------------------
