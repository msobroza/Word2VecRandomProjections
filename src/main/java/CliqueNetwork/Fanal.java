/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package CliqueNetwork;

import java.util.ArrayList;

/**
 *
 * @author msobroza
 */
public class Fanal {
    
    private final int number;
    private final ClusterClique cluster;
    private final ArrayList<Fanal> edges;
    
    public Fanal(int number, ClusterClique cluster){
        this.number = number;
        this.cluster = cluster;
        this.edges = new ArrayList<>();
    }
    
    public void addNonOrientedEdge(Fanal f){
        this.edges.add(f);
    }
    
    public ArrayList<Fanal> getEdges(){
        return this.edges;
    }
    
    public int getNumber(){
        return this.number;
    }
    
    public ClusterClique getCluster(){
        return this.cluster;
    }
}
