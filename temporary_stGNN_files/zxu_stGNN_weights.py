weights = np.zeros((45, 45))
        counts = [3,3,3,3,2,3,3,3,2,3,3,3,3,3,3,2]
        cntr = 0
        ran = [x for x in range(0,counts[cntr])]
        runsran = 0 
        for i in range(len(weights)):
            for j in range(i,len(weights)):
                if i==j:
                    weights[i][j] =0.0 
                elif j in ran and i in ran:
                    weights[i][j] = 1.0
                else: 
                    weights[i][j] = 0.5
            if runsran<len(ran)-1:
                runsran+=1
            else: 
                if (cntr<len(counts)-1):

                    cntr+=1
                    ran = [x for x in range(ran[-1]+1,ran[-1]+1+counts[cntr])]
                    runsran = 0
        for i in range(1,len(weights)):
            for j in range(0,j+1):
                weights[i][j] = weights[j][i]