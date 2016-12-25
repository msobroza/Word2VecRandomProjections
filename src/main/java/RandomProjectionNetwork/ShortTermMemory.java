/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork;

import java.util.ArrayList;

/**
 *
 * @author msobroza
 */
public class ShortTermMemory {

    private final short patternFanalCount[][];
    private final short patternCount[];
    private final int numClusters;
    private final int fanalsPerCluster;

    public ShortTermMemory(int numWords, int numClusters, int fanalsPerCluster) {
        patternFanalCount = new short[numWords][numClusters * fanalsPerCluster];
        patternCount = new short[numWords];
        for (int i = 0; i < numWords; i++) {
            for (int j = 0; j < numClusters * fanalsPerCluster; j++) {
                patternFanalCount[i][j] = 0;
            }
            patternCount[i] = 0;
        }
        this.numClusters = numClusters;
        this.fanalsPerCluster = fanalsPerCluster;
    }

    public ArrayList<Integer> selectWTAIndexesPattern(int patternId, int thresholdDifference) {
        ArrayList<Integer> result = new ArrayList<>();
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            int localWinnerIndex = -1;
            short localWinnerValue = Short.MIN_VALUE;
            boolean conditionThreshold = true;
            for (int idFanal = 0; idFanal < fanalsPerCluster; idFanal++) {
                if (patternFanalCount[patternId][iCluster * fanalsPerCluster + idFanal] > localWinnerValue) {
                    localWinnerValue = patternFanalCount[patternId][iCluster * fanalsPerCluster + idFanal];
                    localWinnerIndex = iCluster * fanalsPerCluster + idFanal;
                }
            }
            for (int idFanal = 0; idFanal < fanalsPerCluster; idFanal++) {
                short diff = (short) (localWinnerValue - (patternFanalCount[patternId][iCluster * fanalsPerCluster + idFanal]));
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

    public ArrayList<Integer> selectWTAIndexesPattern(int patternId, Double thresholdDifference) {
        return selectWTAIndexesPattern(patternId, (int) (thresholdDifference * patternCount[patternId]));
    }

    // Always it has a local winner
    public ArrayList<Integer> selectWTAIndexesPattern(int patternId) {
        return selectWTAIndexesPattern(patternId, 0);
    }

    public void memActivationsGlobalIndex(int patternId, ArrayList<Integer> activationsIndexes) {
        for (int index : activationsIndexes) {
            patternFanalCount[patternId][index]++;
        }
        patternCount[patternId]++;
    }
    
     // Always it has a local winner
    public void memActivationsLocalIndex(int patternId, int[] activationsIndexes) {
        for (int iCluster = 0; iCluster < activationsIndexes.length; iCluster++) {
            patternFanalCount[patternId][iCluster * fanalsPerCluster + activationsIndexes[iCluster]]++;
        }
        patternCount[patternId]++;
    }

}
