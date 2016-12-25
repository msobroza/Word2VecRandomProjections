/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork;

import java.util.ArrayList;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author msobroza
 */
public class RPDenseLayer extends RPLayer {

    protected final INDArray W;
    protected final int inputDimension;
    protected final int fanalsPerCluster;
    protected final int numClusters;

    public RPDenseLayer(int inputDimension, int numClusters, int fanalsPerCluster, float min, float max) {
        this.numClusters = numClusters;
        this.fanalsPerCluster = fanalsPerCluster;
        this.inputDimension = inputDimension;
        this.W = this.createRandomMatrixInitialization(min, max);
    }

    public INDArray getVectorFromCliques(int[] cliqueIndexes) {
        INDArray v = Nd4j.zeros(fanalsPerCluster * numClusters);
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            v.putScalar(iCluster * fanalsPerCluster + cliqueIndexes[iCluster], 1.0);
        }
        return v;
    }

    public INDArray getVectorFromIndexes(ArrayList<Integer> indexes) {
        INDArray v = Nd4j.zeros(fanalsPerCluster * numClusters);
        for (int id : indexes) {
            v.putScalar(id, 1.0);
        }
        return v;
    }

    public INDArray getActivationsMult(INDArray inputVec) {
        return inputVec.mmul(this.W);
    }

    public int[] getCliqueIndexesWTA(INDArray activations) {
        int result[] = new int[numClusters];
        INDArray indexWinners;
        INDArray clusterActivations = activations.reshape(1, numClusters, fanalsPerCluster);
        indexWinners = Nd4j.argMax(clusterActivations, 2);
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            result[iCluster] = indexWinners.getInt(iCluster);
        }
        return result;
    }

    public INDArray createRandomMatrixInitialization(float min, float max) {
        int[] dim = {inputDimension, fanalsPerCluster * numClusters};
        return WeightInitUtil.initWeights(dim, min, max);
    }

    public INDArray getWeigths() {
        return this.W;
    }
}
