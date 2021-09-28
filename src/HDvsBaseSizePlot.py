import matplotlib.pyplot as plt
import json
import calendar
from matplotlib.font_manager import FontProperties

def snapshot_date_parser(date):
    date = str(date)
    month = int(date[2:4])
    month_text = calendar.month_name[month][:3]
    year_str = date[6:]
    return month_text + " " + year_str

current_results = dates_of_datasets = [13032019, 13072019, 13092019, 13112019, 13032020, 13052020, 13072020, 13092020, 13112020, 13012021]

for date in current_results:
    with open("../results/final-results/core-to-HD"+str(date)+".json") as handle:
        core_to_HD = json.load(handle)
    for key in ['left margin', 'right margin', 'loop detected']:
        del core_to_HD[key]

    tuples_sorted = [(int(key), core_to_HD[key]) for key in core_to_HD]
    tuples_sorted.sort(key=lambda tup: tup[0]) # sort by the second element (degree)

    x = [tuples_sorted[i][0] for i in range(len(tuples_sorted))]
    y = [tuples_sorted[i][1] for i in range(len(tuples_sorted))]
    plt.plot(x, y, label = snapshot_date_parser(date))
        
plt.grid(linestyle=':')

fontP = FontProperties()
fontP.set_size('small')    

# # line 1 points
# x1 = [10,20,30]
# y1 = [20,40,10]
# # plotting the line 1 points 
# plt.plot(x1, y1, label = "line 1")
# # line 2 points
# x2 = [10,20,30]
# y2 = [40,10,30]
# # plotting the line 2 points 
# plt.plot(x2, y2, label = "line 2")
plt.xlabel('base size (number of high degree nodes used)')
# Set the y axis label of the current axis.
plt.ylabel('Bound on highway dimension')
# Set a title of the current axes.
plt.title('Base size vs bound on highway dimension')
# show a legend on the plot
# plt.legend(title='title', bbox_to_anchor=(1.05, 1), loc='upper left')#, prop=fontP)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
# Display a figure.
plt.savefig("../results/base-vs-HD.pdf", format="pdf", bbox_inches='tight') 
# plt.show()