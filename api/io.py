def conf_init(conf_path: str) -> dict:
    """Load config file from a path"""
    import yaml
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    # Tushare token
    conf['tushare_token'] = open(conf['tushare_token']).read().strip('\n')
    
    # Path fill
    for k in conf['path'].keys():
        conf['path'][k] = conf['path'][k].format(conf['api_path'])

    return conf