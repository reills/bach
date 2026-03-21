The current setup is wrong at the data-model level.
                                                                                
  You have one scoreXml trying to serve as:                                     
                                                                                
  - the canonical musical truth                                                 
  - the standard-notation document
  - the tablature document                                                      
  - the interaction map source for two different renderers                      
                                                                                
  That is why it keeps drifting and confusing agents.                           
                                                                                
  What’s Broken
  The current frontend splits responsibility across:                            
                                                                                
  - frontend/src/App.tsx                                                        
  - frontend/src/components/ScoreViewer.tsx                                     
  - frontend/src/components/SheetMusicViewer.tsx                                
                                                                                
  And the API only gives one document shape in:

  - frontend/src/api/types.ts                                                   
  - frontend/src/state/types.ts                                                 
                                                                                
  That means:                                                                   
                                                                                
  - Verovio and AlphaTab are interpreting the same guitar XML differently.      
  - The app is switching renderers, not switching views of one explicit render  
    bundle.                                                                     
  - A single eventHitMap is being treated like it applies equally to both views.  - Guitar tab semantics like <staff-details>, <staff-tuning>, and              
    <technical><string/><fret/></technical> leak into the score view.           
                                                                                
  Correct Architecture                                                          
  Use one truth, but not one XML.                                               
                                                                                
  The one truth should be the backend canonical score model. From that one      
  truth, generate two explicit render artifacts for guitar:                     
                                                                                
  - scoreView: standard notation only                                           
  - tabView: tablature only                                                     
                                                                                
  For piano:                                                                    

  - scoreView only                                                              
                                                                                
  So the API shape should become something like:                                
                                                                                
  type RenderView = {                                                           
    xml: string;                                                                
    measureMap?: Record<string, string>;                                        
    eventHitMap?: Record<string, string>;                                       
  };                                                                            
                                                                                
  type ScoreDocumentBundle = {                                                  
    instrumentMode: 'guitar' | 'piano';                                         
    views: {                                                                    
      score: RenderView;                                                        
      tab?: RenderView;                                                         
    };                                                                          
  };                                                                            
                                                                                
  Then frontend state becomes:                                                  
                                                                                
  type LoadedScoreState = {                                                     
    scoreId: string | null;                                                     
    revision: number | null;                                                    
    document: ScoreDocumentBundle | null;                                       
    draftDocument: ScoreDocumentBundle | null;                                  
  };
                                                                                
  Not:                                                                          
                                                                                
  - scoreXml                                                                    
  - draftXml                                                                    
  - one shared map for everything                                               
                                                                                
  Backend Rules                                                                 
  In src/api/render/musicxml.py, split export into two guitar exporters:        
                                                                                
  - canonical_score_to_standard_musicxml(score)                                 
  - canonical_score_to_tab_musicxml(score)                                      
                                                                                
  For guitar score export:                                                      

  - emit normal notation staff                                                  
  - do not emit tab staff metadata                                              
  - do not emit <staff-details>, <staff-lines>6, <staff-tuning>, <capo>         
  - likely do not emit <technical><string/><fret/></technical> in the score view
    if you want clean sheet music                                               

  For guitar tab export:                                                        
                                                                                
  - emit the current tab-capable metadata                                       
  - keep <staff-details>                                                        
  - keep tuning/capo                                                            
  - keep string/fret technical data                                             
                                                                                
  Both must be derived from the same CanonicalScore, same pitches, same         
  durations, same event IDs.                                                    
                                                                                
  That is the key point: one musical source, two render documents.              
                                                                                
  Frontend Rules                                                                
  Do not make one component try to be both renderers.                           
                                                                                
  Use:                                                                          
                                                                                
  - StandardScoreViewer for views.score.xml                                     
  - TabViewer for views.tab.xml                                                 
                                                                                
  And in frontend/src/App.tsx, choose only by explicit document:                
                                                                                
  - guitar + viewTab === 'score' -> document.views.score                        
  - guitar + viewTab === 'tab' -> document.views.tab                            
  - piano -> document.views.score                                               
                                                                                
  No renderer should contain hidden mode-translation logic.                     
                                                                                
  Why Notes Drift Today                                                         
  The likely causes are:                                                        
                                                                                
  - same guitar XML is being parsed differently by Verovio vs AlphaTab          
  - score-view and tab-view hit maps are being treated as identical             
  - tablature-specific MusicXML is being used as if it were clean score notation                                                                                
  If the tab view and score view must stay aligned, each view needs its own     
  eventHitMap. Do not assume renderer ordering is identical.                    
                                                                                
  Implementation Order                                                          
                                                                                
  1. Change API responses to return views.score and views.tab.                  
  2. Change frontend state to store a render bundle, not raw scoreXml.          
  3. Split backend guitar export into separate score/tab exporters.             
  4. Route SheetMusicViewer only to views.score.xml.                            
  5. Route ScoreViewer/AlphaTab only to views.tab.xml.                          
  6. Keep separate per-view maps.                                               
  7. Remove all frontend logic that toggles standard/tab visibility inside one  
     imported document.                                                         
                                                                                
  That is the clean fix. Not another patch.                                     
                                                                                
  If you want, I can do the actual re-architecture next in this order:          
                                                                                
  1. API/state types                                                            
  2. backend dual-view export                                                   
  3. app wiring                                                                 
  4. renderer cleanup