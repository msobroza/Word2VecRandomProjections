/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Tools;

import Word2VecParser.Word2VecParser;
import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.deeplearning4j.clustering.cluster.Point;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author msobroza
 */
public class KMeans {
    //    BaseDistanceFunction, CosineSimilarity, EuclideanDistance, ManhattanDistance

    public enum DistanceFunction {

        CosinusDistance("cosinesimilarity"), SquaredDistance("euclidean"), ManhattanDistance("manhattandistance"), BaseDistanceFunction("basedistancefunction");
        private final String name;

        DistanceFunction(String name) {
            this.name = name;
        }

        public String getName() {
            return this.name;
        }
    }

    private int maxIterationCount;
    private int clusterCount;
    private DistanceFunction distanceFunction;
    private KMeansClustering instance;

    public KMeans(int maxIterationCount, int clusterCount, DistanceFunction distanceFunction) {
        this.maxIterationCount = maxIterationCount;
        this.clusterCount = clusterCount;
        this.distanceFunction = distanceFunction;
        this.instance = KMeansClustering.setup(clusterCount, maxIterationCount, distanceFunction.getName());
    }

    public ArrayList<INDArray> getCentroids(ArrayList<INDArray> vectors) {
        ArrayList<INDArray> result= new ArrayList<>();
        List<Point> pointsLst = Point.toPoints(vectors);
        Word2VecParser.log.info("Start Clustering " + pointsLst.size() + " points");
        ClusterSet cs = instance.applyTo(pointsLst);
        List<Cluster> clsterLst = cs.getClusters();
        for (Cluster c : clsterLst) {
            Point centroids = c.getCenter();
            result.add(centroids.getArray());
        }
        return result;
    }

}
