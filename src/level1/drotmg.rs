#[inline]
pub fn drotmg(sd1: &mut f64, sd2: &mut f64, sx1: &mut f64, sy1: f64, param: &mut [f64; 5]) {
    const GAM: f64 = 4096.0;
    const GAMSQ: f64 = GAM * GAM;
    const RGAMSQ: f64 = 1.0 / GAMSQ;

    // potential params
    let mut sflag: f64;
    let mut sh11: f64 = 0.0;
    let mut sh12: f64 = 0.0;
    let mut sh21: f64 = 0.0;
    let mut sh22: f64 = 0.0;

    // undefined; kill
    if *sd1 < 0.0 {
        sflag = -1.0;
        *sd1 = 0.0;
        *sd2 = 0.0;
        *sx1 = 0.0;
    } else {
        let sp2 = *sd2 * sy1;
        // second component already 0
        if sp2 == 0.0 {
            param[0] = -2.0;
            return;
        }

        let sp1 = *sd1 * *sx1;
        let sq2 = sp2 * sy1;
        let sq1 = sp1 * *sx1;

        if sq1.abs() > sq2.abs() {
            sh21 = -sy1 / *sx1;
            sh12 = sp2 / sp1;
            let su = 1.0 - sh12 * sh21;

            // undefined; kill
            if su <= 0.0 {
                sflag = -1.0;
                *sd1 = 0.0;
                *sd2 = 0.0;
                *sx1 = 0.0;
                sh11 = 0.0;
                sh12 = 0.0;
                sh21 = 0.0;
                sh22 = 0.0;
            } else {
                sflag = 0.0;
                *sd1 /= su;
                *sd2 /= su;
                *sx1 *= su;
            }
        } else {
            // undefined; kill
            if sq2 < 0.0 {
                sflag = -1.0;
                *sd1 = 0.0;
                *sd2 = 0.0;
                *sx1 = 0.0;
                sh11 = 0.0;
                sh12 = 0.0;
                sh21 = 0.0;
                sh22 = 0.0;
            } else {
                sflag = 1.0;
                sh11 = sp1 / sp2;
                sh22 = *sx1 / sy1;
                let su = 1.0 + sh11 * sh22;

                let stemp = *sd2 / su;
                *sd2 = *sd1 / su;
                *sd1 = stemp;
                *sx1 = sy1 * su;
            }
        }

        if *sd1 != 0.0 {
            while *sd1 <= RGAMSQ || *sd1 >= GAMSQ {
                if sflag == 0.0 {
                    sh11 = 1.0;
                    sh22 = 1.0;
                    sflag = -1.0;
                } else {
                    sh21 = -1.0;
                    sh12 = 1.0;
                    sflag = -1.0;
                }

                if *sd1 <= RGAMSQ {
                    *sd1 *= GAMSQ;
                    *sx1 /= GAM;
                    sh11 /= GAM;
                    sh12 /= GAM;
                } else {
                    *sd1 /= GAMSQ;
                    *sx1 *= GAM;
                    sh11 *= GAM;
                    sh12 *= GAM;
                }
            }
        }

        if *sd2 != 0.0 {
            while (*sd2).abs() <= RGAMSQ || (*sd2).abs() >= GAMSQ {
                if sflag == 0.0 {
                    sh11 = 1.0;
                    sh22 = 1.0;
                    sflag = -1.0;
                } else {
                    sh21 = -1.0;
                    sh12 = 1.0;
                    sflag = -1.0;
                }

                if (*sd2).abs() <= RGAMSQ {
                    *sd2 *= GAMSQ;
                    sh21 /= GAM;
                    sh22 /= GAM;
                } else {
                    *sd2 /= GAMSQ;
                    sh21 *= GAM;
                    sh22 *= GAM;
                }
            }
        }
    }

    // pack params
    param[1] = 0.0;
    param[2] = 0.0;
    param[3] = 0.0;
    param[4] = 0.0;

    if sflag < 0.0 {
        param[1] = sh11;
        param[2] = sh21;
        param[3] = sh12;
        param[4] = sh22;
    } else if sflag == 0.0 {
        param[2] = sh21;
        param[3] = sh12;
    } else {
        param[1] = sh11;
        param[4] = sh22;
    }

    param[0] = sflag;
}

