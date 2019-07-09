#include "HookeJeeves.hpp"

// Parameters received:
//   - uint: number of Dimensions
//   - double: rho
//   - double: epsilon
HookeJeeves::HookeJeeves(uint _nd, uint _pl, double _rho, double _e){
  nvars   = _nd;
  PL      = _pl;
  rho     = _rho;
  epsilon = _e;

  printf(" | Number of Dimensions:        %d\n", nvars);
  printf(" | Protein Length:              %d\n", PL);
  printf(" | Rho:                         %.3lf\n", rho);
  printf(" | Epsilon                      %.3lf\n", epsilon);

  startpt = new double[nvars];
  delta   = new double[nvars];
  newx    = new double[nvars];
  xbef    = new double[nvars];
  z       = new double[nvars];

  memset(delta, 0, sizeof(double) * nvars);

  points.reserve(PL);

  if( PL == 13 ){
    AB_SQ = "ABBABBABABBAB";
  } else if( PL == 21 ){
    AB_SQ = "BABABBABABBABBABABBAB";
  } else if( PL == 34 ){
    AB_SQ = "ABBABBABABBABBABABBABABBABBABABBAB";
  } else if( PL == 38 ){
    AB_SQ = "AAAABABABABABAABAABBAAABBABAABBBABABAB";
  } else if( PL == 55 ){
    AB_SQ = "BABABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBAB";
  } else if( PL == 64 ){
    AB_SQ = "ABBABAABBABABBBAABBABABBBABBABABBABABBABABABAABABBAABBABBBAAABAB";
  } else if( PL == 98 ){
    AB_SQ = "AABABAAAAAAABBBAAAAAABAABAABBAABABAAABBBAAAABABAAABABBAAABAAABAAABAABBAABAAAAABAAABABBBABBAAABAABA";
  } else if( PL == 120 ){
    AB_SQ = "ABBABBAABABABAABBAAAABAABABBABABBAAABBBAABBBABAAABABBABBABBBBABBBBAABBBBBBBABABBAAAABBBBBBABBBBAAAABBBABABBBBAAAABBABABB";
  } else {
    std::cout << "Error, AB string string sequence only defined to 13, 21, 34, 38, 55, 64, 98, and 120.\n";
    exit(-1);
  }
}

HookeJeeves::~HookeJeeves(){
  delete [] startpt;
  delete [] delta;
  delete [] newx;
  delete [] xbef;
}

double HookeJeeves::evaluate(double * S){
  points.clear();

	points.push_back( std::make_tuple(0.0, 0.0, 0.0) ); // [0]
	points.push_back( std::make_tuple(0.0, 1.0, 0.0) ); // [1]
  points.push_back( std::make_tuple(cos(S[0]), 1.0 + sin(S[0]), 0.0) ); // [2]

	double _x, _y, _z;
	_x = std::get<0>(points[2]);
	_y = std::get<1>(points[2]);
	_z = std::get<2>(points[2]);

	double * theta = &S[0];
	double * beta  = &S[PL-2];

	for( uint16_t i = 3; i < PL; i++ ){
		_x += cos(theta[i-2])*cos(beta[i-3]);
		_y += sin(theta[i-2])*cos(beta[i-3]);
		_z += sin(beta[i-3]);

		points.push_back(std::make_tuple(_x, _y, _z));
	}

	// printf("Pontos: \n");
	// for( uint16_t i = 0; i < PL; i++ ){
	// 	printf("%.3f %.3f %.3f\n", std::get<0>(points[i]), std::get<1>(points[i]), std::get<2>(points[i]));
	// }

	double v1 = 0.0, v2 = 0.0;
	double xi, xj, yi, yj, zi, zj, dx, dy, dz, D;
	double c_ab;
	for( uint16_t i = 0; i < PL-2; i++ ){
		v1 += 1 - cos(theta[i]);
		for( uint16_t j = i + 2; j < PL; j++ ){
			if (AB_SQ[i] == 'A' && AB_SQ[j] == 'A') //AA bond
				c_ab = 1;
			else if (AB_SQ[i] == 'B' && AB_SQ[j] == 'B') //BB bond
				c_ab = 0.5;
			else
				c_ab = -0.5; //AB or BA bond

			xi = std::get<0>(points[i]);
			xj = std::get<0>(points[j]);

			yi = std::get<1>(points[i]);
			yj = std::get<1>(points[j]);

			zi = std::get<2>(points[i]);
			zj = std::get<2>(points[j]);

			dx = (xi - xj);
			dx *= dx;

			dy = (yi - yj);
			dy *= dy;

			dz = (zi - zj);
			dz *= dz;

			D = sqrt(dx + dy + dz);

			v2 += ( 1 / pow(D, 12) - c_ab / pow(D, 6) );
		}
	}
	// printf("v1: %.4lf v2: %.4lf\n", v1/4, 4*v2);
	// printf("Final energy value: %.8lf\n", v1/4 + 4*v2);
	return(v1/4 + 4*v2);
}

double HookeJeeves::best_nearby(double * point, double prevbest, uint * eval){
  double minf, ftmp;

  // save point on z
  memcpy(z, point, sizeof(double) * nvars);

  minf = prevbest;

  for( uint i = 0; i < nvars; i++ ){
    z[i] = point[i] + delta[i];

    // check bounds
    if( z[i] <= -PI ){
      z[i] += 2.0 * PI;
    } else if(z[i] > PI ){
      z[i] += 2.0 * -PI;
    }

    ftmp = evaluate(z);
    (*eval)++;

    if( ftmp < minf ){
      minf = ftmp;
    } else {
      delta[i] = - delta[i];
      z[i] = point[i] + delta[i];

      // check bounds
      if( z[i] <= -PI ){
        z[i] += 2.0 * PI;
      } else if(z[i] > PI ){
        z[i] += 2.0 * -PI;
      }

      ftmp = evaluate(z);
      (*eval)++;

      if( ftmp < minf )
        minf = ftmp;
      else
        z[i] = point[i];
    }
  }
  memcpy(point, z, sizeof(double) * nvars);

  return minf;
}

double HookeJeeves::optimize(const uint n_evals, double * _startpt){
  bool keep_on;

  memcpy(newx, _startpt, sizeof(double) * nvars);
  memcpy(xbef, _startpt, sizeof(double) * nvars);

  uint it;
  for( it = 0; it < nvars; it++ ){
    delta[it] = fabs(_startpt[it] * rho);
    if( delta[it] == 0.0 )
      delta[it] = rho;
  }

  double fbef;
  double fnew;
  double tmp;


  fbef = evaluate(newx);

  //printf("Entrou com: %.10lf\n", fbef);

  fnew = fbef;

  double step_length = rho;

  it = 0;
  while( it < n_evals && step_length > epsilon ){
    memcpy(newx, xbef, sizeof(double) * nvars);

    fnew = best_nearby(newx, fbef, &it);

    // if we made some improvements, pursue that direction
    keep_on = true;
    while( (fnew < fbef) && (keep_on == true) ){
      for( uint i = 0; i < nvars; i++ ){

        // firstly, arrange the sign of delta[]
        if( newx[i] <= xbef[i] )
          delta[i] = -fabs(delta[i]);
        else
          delta[i] = fabs(delta[i]);

        // now, move further in this direction
        tmp     = xbef[i];
        xbef[i] = newx[i];
        newx[i] = newx[i] + newx[i] - tmp;

        // check bounds
        if( newx[i] <= -PI ){
          newx[i] += 2.0 * PI;
        } else if(newx[i] > PI ){
          newx[i] += 2.0 * -PI;
        }
      }
      fbef = fnew;

      fnew = best_nearby(newx, fbef, &it);

      if( it > n_evals ) break;

      // if the further (optimistic) move was bad
      if( fnew >= fbef ) break;

      keep_on = false;
      for( uint i = 0; i < nvars; i++ ){
        keep_on = true;
        if( fabs(newx[i] - xbef[i]) > (0.5 * fabs(delta[i])) )
          break;
        else
          keep_on = false;
      }
    }
    if( (step_length >= epsilon) and (fnew >= fbef) ){
      step_length *= rho;
      for( uint i = 0; i < nvars; i++ )
        delta[i] *= rho;
    }
  }

  printf(" | HJ uses %d / %d iterations.\n", it, n_evals );
  // copy the improved result to startpt
  for( uint i = 0; i < nvars; i++ )
    _startpt[i] = xbef[i];

  //printf("Saiu com: %.10lf\n", fbef);
  return fbef;
}
