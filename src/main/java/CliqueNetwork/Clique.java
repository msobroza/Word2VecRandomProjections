/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package CliqueNetwork;

import Model.Embedding.Embedding;
import java.util.HashSet;

/**
 *
 * @author msobroza
 */
public class Clique {
    
    private final HashSet<Fanal> fanals;
    private final Embedding embedding;
    
    public Clique(Embedding embedding){
        this.embedding=embedding;
        this.fanals= new HashSet<>();
    }
    
    public Embedding getEmbedding(){
        return this.embedding;
    }
    
    public void addFanal(Fanal fSource){
        // Creates non-oriented edges between fanals
        for(Fanal fDestination:fanals){
            fDestination.addNonOrientedEdge(fSource);
            fSource.addNonOrientedEdge(fDestination);
        }
        this.fanals.add(fSource);
    }
}
