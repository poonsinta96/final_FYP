import copy

def create_compound_arr(a1, a2):
    c_arr = copy.deepcopy(a1)

    for item in a2:
        c_arr.append(item)

    return c_arr


def rules_needed(nested_list):
    #exit conditions
    if nested_list == []:
        return ['']

    cur_list = nested_list[-1]
    new_list = nested_list[:-1]

    ans_arr = []
    for cluster in range(len(cur_list)):

        x = rules_needed(new_list)

        for index in range(len(x)):
            x[index] += str(cluster)
            x[index] += '/'

        ans_arr += x

    #print(ans_arr)
    return ans_arr

def rules_label(ls, label):
    ans = []
    for item in ls:
        name = label + '/' + item
        ans.append(name)

    return ans
