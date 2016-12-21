/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Tools;

import java.util.BitSet;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

/**
 *
 * @author msobroza
 */
public interface BinaryOperations {

    public static final double[] codeWords = {0.0, 1.0};

    public static INDArray binarizeVectorToDoubleArray(double[] vectorD, double threshold) {
        INDArray v = Nd4j.create(vectorD);
        INDArray result = v.cond(new Condition() {
            @Override
            public Boolean apply(Number input) {
                return input.doubleValue() > threshold;
            }

            @Override
            public Boolean apply(IComplexNumber icn) {
                throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
            }
        });
        return result;
    }

    public static BitSet binarizeVectorToBitSet(double[] vectorD, double threshold) {
        BitSet result = new BitSet(vectorD.length);
        for (int i = 0; i < vectorD.length; i++) {
            result.set(i, vectorD[i] > threshold);
        }
        return result;
    }

    public static BitSet[] generateBitHadamardMatrix(int n) {
        BitSet[] hadamard = new BitSet[n];
        hadamard[0].set(0, true);
        for (int k = 1; k < n; k += k) {
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    hadamard[i + k].set(j, hadamard[i].get(j));;
                    hadamard[i].set(j + k, hadamard[i].get(j));
                    hadamard[i + k].set(j + k, !hadamard[i].get(j));
                }
            }
        }
        return hadamard;
    }

    public static double[][] generateDoubleHadamardMatrix(int n) {
        double[][] hadamard = new double[n][n];
        hadamard[0][0] = codeWords[1];
        for (int k = 1; k < n; k += k) {
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    hadamard[i + k][j] = hadamard[i][j];
                    hadamard[i][j + k] = hadamard[i][j];
                    hadamard[i + k][j + k] = getInverseDoubleCodeWord(hadamard[i][j]);
                }
            }
        }
        return hadamard;
    }

    public static double getInverseDoubleCodeWord(double code) {
        if (code == codeWords[0]) {
            return codeWords[1];
        } else {
            return codeWords[0];
        }
    }
}
