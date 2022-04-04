import time

from utils import group_data_loader
import pickle

# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    now = time.time()
    iSub = group_data_loader.load_group_data('iSub')
    noSub = group_data_loader.load_group_data('noSub')
    tSub = group_data_loader.load_group_data('tSub')

    pickle.dump(iSub, open("./pickles/iSub.pkl",'wb'), protocol=2)
    pickle.dump(noSub, open("./pickles/noSub.pkl", 'wb'), protocol=2)
    pickle.dump(tSub, open("./pickles/tSub.pkl", 'wb'), protocol=2)
    #with open("./pickles/noSub.pkl", 'wb') as file:
    #    file.write(pickle.dumps(obj=noSub, protocol=2))
    print(time.time() - now)

    # pickle.dumps(tSub)
    # print(time.time() - now)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
