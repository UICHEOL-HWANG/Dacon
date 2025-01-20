from sklearn.model_selection import train_test_split

def split_data(data):
    train ,test = train_test_split(data, test_size=0.3, random_state=42)
    
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    return train, test 