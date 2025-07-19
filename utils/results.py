def get_mean(scores):
    """
    Returns mean of the scores, ignoring data which is not applicable.
    """
    counts = [ len(scores) for _ in range(len(scores[0])) ]
    totals = [ 0.0 for _ in range(len(scores[0])) ]

    for result in scores:
        for i, value in enumerate(result):
            if value == -1: # Result should be discounted
                counts[i] -= 1
            else:
                totals[i] += value
        
    combined = 0.0 # Combined score for all three landmark classes
    for i, total in enumerate(totals):
        combined += total / counts[i]
        totals[i] = round(total / counts[i], 2)
    combined /= len(totals)
    return totals, round(combined, 2)
    
def write_csv(names, results, path):
    """
    Writes a CSV file containing results of the evaluation.
    """
    open(path, "w+").close() # Overwrites existing file
    
    string = "IMAGE,PREC,DSC,FRCS,CLD\n" # Header
    for i, name in enumerate(names):
        line = f"{name},"
        line += f"{round(results["precision"][i][1], 2)}/{round(results["precision"][i][2], 2)}/{round(results["precision"][i][0], 2)},"
        line += f"{round(results["dsc"][i][1], 2)}/{round(results["dsc"][i][2], 2)}/{round(results["dsc"][i][0], 2)},"
        line += f"{round(results["francois"][i][1], 2)}/{round(results["francois"][i][2], 2)}/{round(results["francois"][i][0], 2)},"
        line += f"{round(results["cld"][i][1], 2)}/{round(results["cld"][i][2], 2)}/{round(results["cld"][i][0], 2)}\n"
        string += line
    
    string += '\n'
    line = "MEAN,"
    mean_prec = get_mean(results["precision"])
    mean_dsc = get_mean(results["dsc"])
    mean_frcs = get_mean(results["francois"])
    mean_cld = get_mean(results["cld"])
    line += f"{mean_prec[0][1]}/{mean_prec[0][2]}/{mean_prec[0][0]},"
    line += f"{mean_dsc[0][1]}/{mean_dsc[0][2]}/{mean_dsc[0][0]},"
    line += f"{mean_frcs[0][1]}/{mean_frcs[0][2]}/{mean_frcs[0][0]},"
    line += f"{mean_cld[0][1]}/{mean_cld[0][2]}/{mean_cld[0][0]}\n"
    string += line
    string += f"OVERALL,{mean_prec[1]},{mean_dsc[1]},{mean_frcs[1]},{mean_cld[1]}"
    
    f = open(path, "w+")
    f.write(string)
    f.close()

def scores_per_landmark(pred, truth, metric):
    """
    Scores a sample for each landmark class in a given metric.
    """
    scores = []
    for i in range(1, 4):
        scores.append(metric(pred, truth, i))
    return scores