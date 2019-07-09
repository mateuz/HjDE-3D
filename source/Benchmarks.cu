#include "Benchmarks.cuh"

Benchmarks::Benchmarks()
{
  min = -100.0;
  max = +100.0;
  n_dim = 100;
  protein_length = 1;

  protein_sequences["1BXP"] = "ABBBBBBABBBAB";
  protein_sequences["1CB3"] = "BABBBAABBAAAB";
  protein_sequences["1BXL"] = "ABAABBAAAAABBABB";
  protein_sequences["1EDP"] = "ABABBAABBBAABBABA";
  protein_sequences["2ZNF"] = "ABABBAABBABAABBABA";
  protein_sequences["1EDN"] = "ABABBAABBBAABBABABAAB";
  protein_sequences["2H3S"] = "AABBAABBBBBABBBABAABBBBBB";
  protein_sequences["1ARE"] = "BBBAABAABBABABBBAABBBBBBBBBBB";
  protein_sequences["2KGU"] = "ABAABBAABABBABAABAABABABABABAAABBB";
  protein_sequences["1TZ4"] = "BABBABBAABBAAABBAABBAABABBBABAABBBBBB";
  protein_sequences["1TZ5"] = "AAABAABAABBABABBAABBBBAABBBABAABBABBB";
  protein_sequences["1AGT"] = "AAAABABABABABAABAABBAAABBABAABBBABABAB";
  protein_sequences["1CRN"] = "BBAAABAAABBBBBAABAAABABAAAABBBAAAAAAAABAAABBAB";
  protein_sequences["2KAP"] = "BBAABBABABABABBABABBBBABAABABAABBBBBBABBBAABAAABBABBABBAAAAB";
  protein_sequences["1HVV"] = "BAABBABBBBBBAABABBBABBABBABABAAAAABBBABAABBABBBABBAABBABBAABBBBBAABBBBBABBB";
  protein_sequences["1GK4"] = "ABABAABABBBBABBBABBABBBBAABAABBBBBAABABBBABBABBBAABBABBBBBAABABAAABABAABBBBAABABBBBA";
  protein_sequences["1PCH"] = "ABBBAAABBBAAABABAABAAABBABBBBBABAAABBBBABABBAABAAAAAABBABBABABABABBABBAABAABBBAABBAAABA";
  protein_sequences["2EWH"] = "AABABAAAAAAABBBAAAAAABAABAABBAABABAAABBBAAABABAAABABBAAABAAABAAABAABBAABAAAAABAAABABBBABBAAABAABA";
}

Benchmarks::~Benchmarks()
{
  /* empty */
}

float Benchmarks::getMin(){
  return min;
}

float Benchmarks::getMax(){
  return max;
}

uint Benchmarks::getID(){
  return ID;
}

void Benchmarks::setMin( float _min ){
  min = _min;
}

void Benchmarks::setMax( float _max ){
  max = _max;
}

void Benchmarks::setThreads( uint _n){
  NT.x = _n;
}

void Benchmarks::setBlocks( uint _n ){
  NB.x = _n;
}

uint Benchmarks::getThreads(){
  return NT.x;
}

uint Benchmarks::getBlocks(){
  return NB.x;
}

void Benchmarks::showSequences(){
  // show content:
  printf(" +==============================================================+ \n");
  printf(" |                         PROTEIN SEQUENCES                    | \n");
  printf(" +==============================================================+ \n");

  std::map<std::string,std::string>::iterator it;
  for( it = protein_sequences.begin(); it!=protein_sequences.end(); ++it )
    printf("%s | %s\n", it->first.c_str(), it->second.c_str());

  printf(" +==============================================================+ \n");
}

size_t Benchmarks::findSequence( std::string _w ){
  std::map<std::string, std::string>::iterator it;

  it = protein_sequences.find(_w);
  if(it != protein_sequences.end())
    return it->second.size();

  return 0;
}
