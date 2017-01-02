/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork.DifferentialLayer;

import RandomProjectionNetwork.STM.STMSoft;
import java.util.ArrayList;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author msobroza
 */
public class RPDifferentialSoftLayer extends DifferentialLayer {

    private final STMSoft stm;

    public RPDifferentialSoftLayer(int inputDimension, int numClusters, int fanalsPerCluster, float min, float max) {
        super(inputDimension, numClusters, fanalsPerCluster, min, max);
        this.stm = null;
    }

    public RPDifferentialSoftLayer(int inputDimension, int numClusters, int fanalsPerCluster, int numWords, float min, float max) {
        super(inputDimension, numClusters, fanalsPerCluster, min, max);
        this.stm = new STMSoft(numWords, numClusters, fanalsPerCluster);
    }

    public INDArray getBinaryVectorSTM(int idPattern) {
        return this.getVectorFromIndexes(this.stm.getLWTAPattern(idPattern));
    }

    public INDArray getBinaryVectorSTM(int idPattern, Double thresholdSTM) {
        return this.getVectorFromIndexes(this.stm.getWTAPattern(idPattern, thresholdSTM));
    }

    public INDArray getBinaryVectorLowestClustersSTM(INDArray activationsPattern, int idPattern, Double thresholdSTM) {
        ArrayList<Integer> indexesResult = new ArrayList<>();
        ArrayList<Integer> idLowestClusters = getLowerDifferentialSoftActivationsClusters(activationsPattern, thresholdSTM);
        ArrayList<Integer> indexSTMWordPattern = this.stm.getLWTAPattern(idPattern);
        int[] indexLWTAWordPattern = getCliqueLocalWTA(activationsPattern);
        for (int idCluster = 0; idCluster < this.numClusters; idCluster++) {
            if (idLowestClusters.contains(idCluster)) {
                indexesResult.add(indexSTMWordPattern.get(idCluster));
            } else {
                indexesResult.add(indexLWTAWordPattern[idCluster]);
            }
        }
        return this.getVectorFromIndexes(indexesResult);
    }

    public void memorizeSampleGlobalWTA(INDArray activationsSample, Double thresholdDifferential, int idPattern) {
        INDArray indexWinners = getArgMaxIndexes(activationsSample);
        INDArray diffDivMax = getDivDiffMax(activationsSample, this.numClusters, this.fanalsPerCluster);
        ArrayList<Integer> globalIndexesClique = new ArrayList<>();
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            if (diffDivMax.getDouble(iCluster) >= thresholdDifferential) {
                int idFanal = iCluster * fanalsPerCluster + indexWinners.getInt(iCluster);
                globalIndexesClique.add(idFanal);
            }
        }
        this.stm.memActivationsPattern(idPattern, getVectorFromIndexes(globalIndexesClique));
    }

    public void memorizeSampleSoft(INDArray activationsSample, int idPattern) {
        this.stm.memActivationsPattern(idPattern, activationsSample);
    }

    public INDArray getVectorDifferentialWTA(INDArray activations, Double threshold) {
        INDArray indexWinners = getArgMaxIndexes(activations);
        INDArray diff = getDiffMaxMin(activations, this.numClusters, this.fanalsPerCluster);
        ArrayList<Integer> resultIndexes = new ArrayList<>();
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            if (diff.getDouble(iCluster) >= threshold) {
                int idFanal = iCluster * fanalsPerCluster + indexWinners.getInt(iCluster);
                resultIndexes.add(idFanal);
            }
        }
        return this.getVectorFromIndexes(resultIndexes);
    }

    public ArrayList<Integer> getIndexesDifferentialSoftWTA(INDArray activations, Double threshold) {
        INDArray softmaxResult;
        INDArray indexWinners = getArgMaxIndexes(activations);
        INDArray input = activations.reshape(numClusters, fanalsPerCluster);
        softmaxResult = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", input));
        INDArray diff = getDiffMaxMin(softmaxResult, this.numClusters, this.fanalsPerCluster);
        ArrayList<Integer> resultIndexes = new ArrayList<>();
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            if (diff.getDouble(iCluster) >= threshold) {
                int idFanal = iCluster * fanalsPerCluster + indexWinners.getInt(iCluster);
                resultIndexes.add(idFanal);
            }
        }
        return resultIndexes;
    }

    public ArrayList<Integer> getLowerDifferentialSoftActivationsClusters(INDArray activations, Double threshold) {
        INDArray softmaxResult;
        INDArray input = activations.reshape(numClusters, fanalsPerCluster);
        softmaxResult = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", input));
        INDArray diff = getDiffMaxMin(softmaxResult, this.numClusters, this.fanalsPerCluster);
        ArrayList<Integer> resultIndexes = new ArrayList<>();
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            if (diff.getDouble(iCluster) < threshold) {
                resultIndexes.add(iCluster);
            }
        }
        return resultIndexes;
    }

    public INDArray getVectorDifferentialSoftWTA(INDArray activations, Double threshold) {
        return this.getVectorFromIndexes(getIndexesDifferentialSoftWTA(activations, threshold));
    }

    public INDArray getVectorLWTA(INDArray activations) {
        return getVectorFromCliques(getCliqueLocalWTA(activations));
    }

}
