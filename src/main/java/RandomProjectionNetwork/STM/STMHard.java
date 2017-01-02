/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork.STM;

import RandomProjectionNetwork.STM.ShortTermMemory;
import java.util.ArrayList;

/**
 *
 * @author msobroza
 */
public class STMHard extends ShortTermMemory {

    private final short[][] patternFanalAcc;

    public STMHard(int numWords, int numClusters, int fanalsPerCluster) {
        super(numWords, numClusters, fanalsPerCluster);
        patternFanalAcc = new short[numWords][numClusters * fanalsPerCluster];
        for (int i = 0; i < numWords; i++) {
            for (int j = 0; j < numClusters * fanalsPerCluster; j++) {
                patternFanalAcc[i][j] = 0;
            }
        }
    }

    public ArrayList<Integer> getWTAPattern(int patternId, int thresholdDifference) {
        ArrayList<Integer> result = new ArrayList<>();
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            int localWinnerIndex = -1;
            short localWinnerValue = Short.MIN_VALUE;
            boolean conditionThreshold = true;
            for (int idFanal = 0; idFanal < fanalsPerCluster; idFanal++) {
                if (patternFanalAcc[patternId][iCluster * fanalsPerCluster + idFanal] > localWinnerValue) {
                    localWinnerValue = patternFanalAcc[patternId][iCluster * fanalsPerCluster + idFanal];
                    localWinnerIndex = iCluster * fanalsPerCluster + idFanal;
                }
            }
            for (int idFanal = 0; idFanal < fanalsPerCluster; idFanal++) {
                short diff = (short) (localWinnerValue - (patternFanalAcc[patternId][iCluster * fanalsPerCluster + idFanal]));
                if (diff <= thresholdDifference && localWinnerIndex != iCluster * fanalsPerCluster + idFanal) {
                    conditionThreshold = false;
                }
            }
            if (conditionThreshold) {
                result.add(localWinnerIndex);
            }
        }
        return result;
    }

    public ArrayList<Integer> getWTAPattern(int patternId, Double thresholdDifference) {
        return STMHard.this.getWTAPattern(patternId, (int) (thresholdDifference * patternAcc[patternId]));
    }

    // Always it has a local winner
    public ArrayList<Integer> getWTAPattern(int patternId) {
        return STMHard.this.getWTAPattern(patternId, 0);
    }

    public void memActivationsGlobal(int patternId, ArrayList<Integer> activationsIndexes) {
        for (int index : activationsIndexes) {
            patternFanalAcc[patternId][index]++;
        }
        patternAcc[patternId]++;
    }

    // Always it has a local winner
    public void memActivationsLocal(int patternId, int[] activationsIndexes) {
        for (int iCluster = 0; iCluster < activationsIndexes.length; iCluster++) {
            patternFanalAcc[patternId][iCluster * fanalsPerCluster + activationsIndexes[iCluster]]++;
        }
        patternAcc[patternId]++;
    }

}
