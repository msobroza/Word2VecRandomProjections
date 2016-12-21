/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package CliqueNetwork;

import Model.Anchor.Anchor;
import Model.Anchor.Centroid;
import Model.Anchor.AnchorDistribution;
import Model.CodingNeighbors.CodeHadamard;
import Model.Embedding.Embedding;
import Model.Embedding.EmbeddingReduce;
import Model.Anchor.AnchorRandomCode;
import Model.CodingNeighbors.CodeDistribution;
import Model.Embedding.EmbeddingHadamard;
import Tools.BinaryOperations;
import static Tools.BinaryOperations.generateBitHadamardMatrix;
import Model.CodingNeighbors.CodeHadamardBit;
import Model.Embedding.EmbeddingBit;
import Model.Embedding.EmbeddingDouble;
import Model.Embedding.EmbeddingHadamardBit;
import Model.Range;
import Word2VecParser.Config.MethodAnchorPoints;
import Tools.KMeans;
import static Word2VecParser.Word2VecParser.ACTIVATION_VALUE;
import static Word2VecParser.Word2VecParser.CODE_LENGTH;
import static Word2VecParser.Word2VecParser.FANALS_PER_CLUSTER;
import static Word2VecParser.Word2VecParser.GAMA;
import static Word2VecParser.Word2VecParser.KMEANS_DISTANCE_FUNCTION;
import static Word2VecParser.Word2VecParser.KMEANS_MAX_ITERATION_COUNT;
import static Word2VecParser.Word2VecParser.NUMBER_RANGES;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Comparator;
import org.nd4j.linalg.api.ndarray.INDArray;
import static Word2VecParser.Word2VecParser.NUMBER_ANCHOR_POINTS;
import static Word2VecParser.Word2VecParser.METHOD_ANCHOR_POINTS;
import static Word2VecParser.Word2VecParser.NUMBER_BINARY_SELECTED_CODES;
import static Word2VecParser.Word2VecParser.NUMBER_BINARY_CODES;
import static Word2VecParser.Word2VecParser.NUMBER_DISTRIBUTIONS;
import static Word2VecParser.Word2VecParser.N_HADAMARD;
import java.util.BitSet;
import java.util.List;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author msobroza
 */
class CompareEmbeddingL1Norm implements Comparator<EmbeddingReduce> {

    @Override
    public int compare(EmbeddingReduce o1, EmbeddingReduce o2) {
        return (o1.getL1Norm0() < o2.getL1Norm0() ? -1 : (o1.getL1Norm0() == o2.getL1Norm0() ? 0 : 1));
    }
}

class CompareEmbeddingSum implements Comparator<EmbeddingReduce> {

    @Override
    public int compare(EmbeddingReduce o1, EmbeddingReduce o2) {
        return (o1.getAbsSumComponents() < o2.getAbsSumComponents() ? -1 : (o1.getAbsSumComponents() == o2.getAbsSumComponents() ? 0 : 1));
    }
}

class CompareCodeBySumIntervals implements Comparator<AnchorRandomCode> {

    @Override
    public int compare(AnchorRandomCode o1, AnchorRandomCode o2) {
        return (o1.getSumIntervals() < o2.getSumIntervals() ? -1 : (o1.getSumIntervals() == o2.getSumIntervals() ? 0 : 1));
    }
}

public class CliqueNet {

    private ArrayList<ClusterClique> clusters;
    private HashMap<Fanal, ClusterClique> mapFanalCluster;
    private ArrayList<Embedding> anchors;
    private INDArray W;
    private int fanalsPerCluster;
    private int numClusters;

    public CliqueNet() {
        this.anchors = new ArrayList<>();
        this.W = null;
        this.numClusters=0;
        this.fanalsPerCluster=0;
    }

    public void createRandomMatrixInitialization(int dimensionEmbedding, int fanalsPerCluster, int numClusters, float min, float max) {
        int[] dim = {dimensionEmbedding, fanalsPerCluster*numClusters};
        this.W = WeightInitUtil.initWeights(dim, min, max);
        this.numClusters=numClusters;
        this.fanalsPerCluster=fanalsPerCluster;
    }
    
    public int[] getCliqueFromRandomMultiplicationMatrix(String word, INDArray vec){
        INDArray activations = vec.mmul(this.W);
        int result [] = new int[numClusters];
        INDArray indexWinners;
        INDArray clusterActivations=activations.reshape(1, numClusters, fanalsPerCluster);
        indexWinners = Nd4j.argMax(clusterActivations, 2);
        for(int iCluster=0; iCluster<numClusters;iCluster++){
            result[iCluster]= indexWinners.getInt(iCluster);
        }
        return result;
    }

    public void createRandomAnchorPoints(ArrayList<EmbeddingDouble> embVector) {
        // Sets clusters and centroids;
        for (int i = 0; i < NUMBER_ANCHOR_POINTS; i++) {
            // Select random embedding for centroids
            int item = new Random().nextInt(embVector.size());
            while (isAnchorPoint(embVector.get(item))) {
                item = new Random().nextInt(embVector.size());
            }
            this.anchors.add(i, new AnchorDistribution(embVector.get(item), embVector));
            System.out.println("Anchor point: " + i + " , Word: " + this.anchors.get(i).getWord());
        }
    }

    public void createRandomBinaryCodes(ArrayList<EmbeddingDouble> embVector) {
        List<AnchorRandomCode> lstCodes = new ArrayList<>();
        for (int i = 0; i < NUMBER_BINARY_CODES; i++) {
            lstCodes.add(new AnchorRandomCode(i, CODE_LENGTH, embVector));
        }
        Collections.sort(lstCodes, new CompareCodeBySumIntervals());
        // Selecting best codes maximize sum intervals
        List<AnchorRandomCode> bestCodes = lstCodes.subList(lstCodes.size() - NUMBER_BINARY_SELECTED_CODES - 1, lstCodes.size());
        this.anchors.addAll(bestCodes);
        for (AnchorRandomCode c : bestCodes) {
            System.out.println("Num code: " + c.getNumCode() + " Sum intervals: " + c.getSumIntervals());
        }
    }

    public ArrayList<EmbeddingHadamardBit> createDistributedHadamardCodes(ArrayList<EmbeddingBit> embVector) {
        ArrayList<CodeHadamardBit> lstCodes = new ArrayList<>();
        BitSet[] hadamard = generateBitHadamardMatrix(N_HADAMARD);
        for (int i = 0; i < N_HADAMARD; i++) {
            lstCodes.add(new CodeHadamardBit(i, hadamard[i]));
        }
        for (int j = 0; j < NUMBER_DISTRIBUTIONS; j++) {
            this.anchors.add(new CodeDistribution(lstCodes));
        }
        ArrayList<EmbeddingHadamardBit> lstEmbHadamard = new ArrayList<>();
        for (EmbeddingBit e : embVector) {
            lstEmbHadamard.add(new EmbeddingHadamardBit(e, lstCodes));
        }
        for (int k = 0; k < this.anchors.size(); k++) {
            for (EmbeddingHadamardBit eh : lstEmbHadamard) {
                ((CodeDistribution) this.anchors.get(k)).addEmbedding(eh, k);
            }
        }
        return lstEmbHadamard;
    }

    public void createKMeansCentroidsAnchorPoints(ArrayList<EmbeddingDouble> embVector) {
        KMeans kMeansInstance = new KMeans(KMEANS_MAX_ITERATION_COUNT, NUMBER_ANCHOR_POINTS, KMEANS_DISTANCE_FUNCTION);
        ArrayList<INDArray> inputVectors = new ArrayList<>();
        embVector.stream().forEach((e) -> {
            inputVectors.add((INDArray) e.getVector());
        });
        ArrayList<INDArray> centroidVectors = kMeansInstance.getCentroids(inputVectors);
        for (int idCentroid = 0; idCentroid < centroidVectors.size(); idCentroid++) {
            Centroid c = new Centroid(idCentroid, centroidVectors.get(idCentroid));
            this.anchors.add(idCentroid, new AnchorDistribution((EmbeddingDouble) c, embVector));
            System.out.println("Anchor point: " + idCentroid + " , Word: " + this.anchors.get(idCentroid).getWord());
        }
    }

    public void createSumBasedAnchorPoints(ArrayList<EmbeddingReduce> embVector) {
        if (METHOD_ANCHOR_POINTS.getIndex() == MethodAnchorPoints.Sum.getIndex()) {
            Collections.sort(embVector, new CompareEmbeddingSum());
        } else {
            Collections.sort(embVector, new CompareEmbeddingL1Norm());
        }
        ArrayList<EmbeddingDouble> vector = new ArrayList<>();
        vector.addAll(embVector);
        for (int i = 0; i < NUMBER_ANCHOR_POINTS; i++) {
            int item = (NUMBER_RANGES / NUMBER_ANCHOR_POINTS) * i;
            EmbeddingReduce embRef = embVector.get(item);
            System.out.println("Item: " + item + " anchor word: " + embRef.getWord() + " Modulo sum: " + embRef.getAbsSumComponents() + " L1 norm: " + embRef.getL1Norm0());
            this.anchors.add(i, new AnchorDistribution((EmbeddingDouble) embRef, vector));
        }
        System.out.println("Number of anchor points: " + this.anchors.size());
    }

    public void createNet(ArrayList<EmbeddingDouble> embVector) {
        createRandomAnchorPoints(embVector);
        this.clusters = new ArrayList<>();
        this.mapFanalCluster = new HashMap<>();
        // Sets clusters and centroids;
        for (int i = 0; i < NUMBER_ANCHOR_POINTS; i++) {
            // Creates a new cluster and fanals
            this.clusters.add(i, new ClusterClique((Anchor) this.anchors.get(i), i * (FANALS_PER_CLUSTER), (i + 1) * (FANALS_PER_CLUSTER - 1)));
        }

        for (int j = 0; j < embVector.size(); j++) {
            Embedding e = embVector.get(j);
            Clique clique = new Clique(e);
            for (int i = 0; i < NUMBER_ANCHOR_POINTS; i++) {
                // Associates a embedding to a fanal
                ClusterClique c = this.clusters.get(i);
                Fanal f = c.mapRandomFanal(e);
                this.mapFanalCluster.put(f, c);
                clique.addFanal(f);
            }
        }
    }

    private boolean isAnchorPoint(Embedding e) {
        return anchors.stream().anyMatch((c) -> (e.equals(c)));
    }

    public void SoS(ArrayList<Fanal> activatedFanals) {
        for (Fanal fprop : activatedFanals) {
            fprop.getCluster().activateFanal(fprop, GAMA);
            for (Fanal fconn : fprop.getEdges()) {
                fconn.getCluster().activateFanal(fconn, ACTIVATION_VALUE);
            }
        }
    }

    public ArrayList<String> decoderAnchorPoint(Embedding question) {
        ArrayList<String> result = new ArrayList<>();
        ArrayList<Range> decoderRanges = new ArrayList<>();
        HashSet<Embedding> intersection = new HashSet<>();
        int idCluster = 0;
        for (Embedding pe : anchors) {
            Anchor p = (Anchor) pe;
            decoderRanges.add(p.getRange(question));
            if (idCluster == 0) {
                intersection.addAll(decoderRanges.get(idCluster).getVectors());
            } else {
                intersection.retainAll(decoderRanges.get(idCluster).getVectors());
            }
            idCluster++;
        }
        for (Embedding e : intersection) {
            result.add(e.getWord());
            //System.out.println("<"+e.getWord()+">");
        }
        return result;
    }

    public ArrayList<String> decoderFuzzyCodeHadamard(EmbeddingHadamardBit question, int n) {
        ArrayList<String> result = new ArrayList<>();
        HashMap<Integer, HashSet<Range>> decoderRanges = new HashMap<>();
        HashSet<Range> rangesCluster;
        HashSet<Embedding> intersection = new HashSet<>();
        HashSet<Embedding> auxVectors;
        int idCluster = 0;
        for (Embedding pe : anchors) {
            CodeDistribution distCode = (CodeDistribution) pe;
            rangesCluster = new HashSet<>();
            rangesCluster.addAll(distCode.getFuzzyRange(question, n));
            //System.out.println("IdCluster: " + idCluster + " , Number of win: " + n + " , Number of ranges: " + rangesCluster.size());
            decoderRanges.put(idCluster, rangesCluster);
            auxVectors = new HashSet<>();
            for (Range r : decoderRanges.get(idCluster)) {
                auxVectors.addAll(r.getVectors());
            }
            if (idCluster == 0) {
                intersection.addAll(auxVectors);
            } else {
                intersection.retainAll(auxVectors);
            }
            //System.out.println("Number of vectors: " + intersection.size());
            idCluster++;
        }
        for (Embedding e : intersection) {
            result.add(e.getWord());
            //System.out.println("<"+e.getWord()+">");
        }
        return result;
    }

    public ArrayList<String> decoderFuzzyAnchorPoint(Embedding question, int n) {
        ArrayList<String> result = new ArrayList<>();
        HashMap<Integer, HashSet<Range>> decoderRanges = new HashMap<>();
        HashSet<Range> rangesCluster;
        HashSet<Embedding> intersection = new HashSet<>();
        HashSet<Embedding> auxVectors;
        int idCluster = 0;
        for (Embedding pe : anchors) {
            Anchor p = (Anchor) pe;
            rangesCluster = new HashSet<>();
            rangesCluster.addAll(p.getFuzzyRange(question, n));
            //System.out.println("IdCluster: " + idCluster + " , Number of win: " + n + " , Number of ranges: " + rangesCluster.size());
            decoderRanges.put(idCluster, rangesCluster);
            auxVectors = new HashSet<>();
            for (Range r : decoderRanges.get(idCluster)) {
                auxVectors.addAll(r.getVectors());
            }
            if (idCluster == 0) {
                intersection.addAll(auxVectors);
            } else {
                intersection.retainAll(auxVectors);
            }
            //System.out.println("Number of vectors: " + intersection.size());
            idCluster++;
        }
        for (Embedding e : intersection) {
            result.add(e.getWord());
            //System.out.println("<"+e.getWord()+">");
        }
        return result;
    }

    public ArrayList<Fanal> WTA() {
        ArrayList<Fanal> result = new ArrayList<>();
        for (ClusterClique c : clusters) {
            result.addAll(c.LWsTA());
        }
        return result;
    }

    public void initScoreAllClusters() {
        for (ClusterClique c : clusters) {
            c.initScore();
        }
    }

}
