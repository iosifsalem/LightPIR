import json
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import ast

def snapshot_date_parser(date):
    date = str(date)
    month = int(date[2:4])
    month_text = calendar.month_name[month][:3]
    year_str = date[6:]
    return month_text + year_str

current_results = [13032019, 13072019, 13092019, 13112019, 13032020, 13052020, 13072020, 13092020, 13112020, 13012021]

results = {}

for date in current_results:
    results[date] = {}
    
    # compute results[date]['hub_sizes']
    with open('../results/final-results/lightning'+str(date)+'-large-component-hubs.JSON') as handle:
        hub_dict = json.load(handle)
    num_of_high_deg_nodes = len(ast.literal_eval(hub_dict["high_deg_nodes"]))
    print(f"for date {date}, num_of_high_deg_nodes = {num_of_high_deg_nodes}")
    del hub_dict["high_deg_nodes"]
    del hub_dict['node_int_to_pkey']    
    
    for node in hub_dict:
        if hub_dict[node] == 'set()':
            hub_dict[node] = []
        else:
            hub_dict[node] = ast.literal_eval(hub_dict[node])
    
    results[date]['hub_sizes'] = [num_of_high_deg_nodes + len(hub_dict[u]) for u in hub_dict]

# make box plot
hub_size_data = {snapshot_date_parser(date):results[date]['hub_sizes'] for date in current_results}

# df = pd.DataFrame(data=hub_size_data)
# df = pd.DataFrame.from_dict(hub_size_data, orient='index')
df = pd.DataFrame({ key:pd.Series(value) for key, value in hub_size_data.items() })
myFig = plt.figure();
df.plot.box(grid='True')
plt.ylabel('hub sizes (number of nodes)')
plt.title('Distributions of hub sizes')
plt.savefig("../results/boxplot-hub-distr-all-data.pdf", format="pdf") 