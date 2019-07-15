#ifndef _BENCHMARKS_H
#define _BENCHMARKS_H

#include <string>
#include <map>

class Benchmarks
{
protected:
  float min;
  float max;

  uint ID;
  uint n_dim;
  uint ps;
  uint protein_length;

  dim3 NT;
  dim3 NB;

  std::map< std::string, std::string> protein_sequences;

public:

  Benchmarks();
  virtual ~Benchmarks();

  virtual void compute(float * x, float * fitness)
  {
    /* empty */
  };

  float getMin();
  float getMax();
  uint getID();

  void setMin( float );
  void setMax( float );

  /* GPU launch compute status */
  void setThreads( uint );
  void setBlocks( uint );

  uint getThreads();
  uint getBlocks();

  void showSequences(void);

  size_t findSequence( std::string );

  std::string getSequence( std::string );
};

#endif
