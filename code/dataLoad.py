import pandas as pd


def dataset_from_name(dataset_name):
    """
    Binarize data given dataset name.
    :param dataset_name:
    :return: pandas.DataFrame
    """
    if dataset_name == 'compas':
        compas = pd.read_csv("./data/compas.data", sep=",")
        compas = compas[(compas.race == "Caucasian") | (compas.race == "African-American")]
        drugs = compas["r_charge_desc"].apply(lambda x: 1. if "Cocaine" in str(x) or "Cannabis" in str(x) else 0.)
        compas = compas[["age", "race", "is_recid", "sex", "r_days_from_arrest", "c_jail_in", "c_jail_out"]]
        compas["c_jail_in"] = pd.to_datetime(compas["c_jail_in"])
        compas["c_jail_out"] = pd.to_datetime(compas["c_jail_out"])
        compas["sentence_time"] = compas["c_jail_out"] - compas["c_jail_in"]
        compas["< 25"] = (compas["age"] < 25)
        compas["> 45"] = (compas["age"] > 45)
        sex = compas["sex"].apply(lambda x: 1. if x == "Male" else 0.)
        young = compas["< 25"].apply(lambda x: 1. if x == True else 0.)
        old = compas["> 45"].apply(lambda x: 1. if x == True else 0.)
        sentence_length = compas["sentence_time"].apply(lambda x: x.days)

        long_sentence = sentence_length.apply(lambda x: 1. if x > 30 else 0.)
        race = compas["race"].apply(lambda x: 1. if x == "Caucasian" else 0.)
        is_recid = compas["is_recid"]

        bin_df = pd.DataFrame()
        bin_df["sex"] = sex
        bin_df["young"] = young
        bin_df["old"] = old
        bin_df["long sentence"] = long_sentence
        bin_df["drugs"] = drugs
        bin_df["race"] = race
        bin_df["recidivism"] = is_recid
        return bin_df
    elif dataset_name == 'adult':
        adult = pd.read_csv("./data/adult.data",
                            names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                                   "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                                   "hours-per-week", "native-country", "income"])
        adult = adult[
            ["age", "workclass", "education", "marital-status", "relationship", "race", "hours-per-week", "sex",
             "native-country", "income"]]
        bin_df = pd.DataFrame()
        bin_df["sex"] = adult["sex"].apply(lambda x: 1. if x.strip() == "Male" else 0.)
        bin_df["age"] = adult["age"].apply(lambda x: 1. if x > 35 else 0.)
        bin_df["native_country"] = adult["native-country"].apply(
            lambda x: 1. if x.strip() in ["United-States", "England", "Canada", "Germany", "Japan", "Italy",
                                          "France"] else 0.)
        bin_df["hours_per_week"] = adult["hours-per-week"].apply(lambda x: 1. if x > 35 else 0.)
        bin_df["education"] = adult["education"].apply(
            lambda x: 1. if x.strip() in ["Bachelors", "Masters", "Doctorate"] else 0.)
        bin_df["relationship"] = adult["relationship"].apply(lambda x: 1. if x.strip() in ["Wife", "Husband"] else 0.)
        bin_df["income"] = adult["income"].apply(lambda x: 1. if x.strip() == ">50K" else 0.)
        return bin_df
    elif dataset_name == 'german':
        german = pd.read_csv("./data/german.data", sep=" ", names=range(1, 22))
        bin_df = pd.DataFrame()
        bin_df["job"] = german[17].apply(lambda x: 1. if x.strip() in ["A173", "A174"] else 0.)
        bin_df["housing"] = german[15].apply(lambda x: 1. if x.strip() in ["A152"] else 0.)
        bin_df["sex"] = german[9].apply(lambda x: 1. if x.strip() in ["A91", "A93", "A94"] else 0.)
        bin_df["savings"] = german[6].apply(lambda x: 1. if x.strip() in ["A63", "A64"] else 0.)
        bin_df["credit_history"] = german[3].apply(lambda x: 1. if x.strip() in ["A30", "A31", "A32"] else 0.)
        bin_df["age"] = german[13].apply(lambda x: 1. if x > 50 else 0.)
        bin_df["returns"] = german[21].apply(lambda x: 1. if x in [1] else 0.)
        return bin_df
    else:
        print('dataset_name is invalid')
        return None
