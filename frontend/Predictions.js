// import logo from './logo.svg';
import './Predictions.css';
import React, { createElement } from 'react';
import { render } from "react-dom";
import f1toycar from '../../images/f1-toy-car.png';

// import components
import Footer from '../footer.js';


// neural net
// xgboost
// xgb classifier
// xgb ranker
// random forest classifier

// External CSS imports
import 'bootstrap/dist/css/bootstrap.css';

import BetsMoney from "../linegraph.js";

// checkmark hex: &#x2714;
// ex hex: &#x2718;

function Predictions() {
    return (
        <>
        <div className = "predictions-container reset">
            <div className = "piece">
                <h1 className = "header-text center-text glow-text">Predictions</h1>
                <div className = "loop-wrapper">
                    <div className = "car" id = "car">
                        <img src = {f1toycar}></img>
                    </div>
                </div>
            </div>

            <div className = "preds center piece">

                <div className = "divide-red"></div>

                <div className = "preds-h3 intro">
                    <h2 className = "center-text">Our Predictions for the Next Race:</h2>
                    <h3 className = "center-text">We have multiple models, and here's our models' outputs for the upcoming race!</h3>

                    <h1 className = "center-text glow-text prediction-output">racer name</h1>
                    <h1 className = "center-text glow-text prediction-output">racer name</h1>
                    <h1 className = "center-text glow-text prediction-output">racer name</h1>
                </div>


                <div className = "divide-red"></div>

            </div>

            <div className = "preds-h3 preds-performance center">
                <h2>Predictions Performance</h2>
                <h3 className = "preds-h3 less-space">How have we been doing? To quantify our performance, we've created some silly metrics to see how we are doing!
                We've gathered data from Travis (a very knowledgeable man), and we compare
                our predictions to his against the actual race results.</h3>

                <table className = "center preds-table">
                    <tr className = "first-row">
                        <th>Race #</th>
                        <th>F1 ML Team (Kelly Strat)</th>
                        <th>F1 ML Team (Naive Strat)</th>
                        <th>Travis</th>
                        <th>Results</th>
                    </tr>
                    <tr>
                        <td>1</td>
                        <td>
                            <em className = "wrong">&#x2718; </em>
                                Perez
                        </td>
                        <td>
                            <em className = "correct">&#x2714; </em>
                                Verstappen
                        </td>
                        <td>
                            <em className = "correct">&#x2714; </em>
                                Verstappen
                        </td>
                        <td>Verstappen</td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td>
                            <em className = "correct">&#x2714; </em>
                                Perez
                        </td>
                        <td>
                            <em className = "correct">&#x2714; </em>
                                Perez
                        </td>
                        <td>
                            <em className = "wrong">&#x2718; </em>
                            Verstappen,
                            Alonso (podium)
                        </td>
                        <td>Perez</td>
                    </tr>
                    <tr>
                        <td>3</td>
                        <td>
                            <em className = "correct">&#x2714; </em>
                            Verstappen (1st), Hamilton (2nd)
                            </td>
                        <td>
                            <em className = "correct">&#x2714; </em>
                            Verstappen
                        </td>
                        <td>
                            <em className = "correct">&#x2714; </em>
                            Verstappen
                        </td>
                            <td>Verstappen</td>
                    </tr>
                    <tr>
                        <td>4</td>
                        <td>
                            <em className = "symm">&#x2714; fix </em>
                            perez
                        </td>
                        <td>
                            <em className = "symm">&#x2714; fix </em>
                            perez
                        </td>
                        <td>
                            <em className = "sym">&#x2718; fix </em>
                            verstappen
                        </td>
                            <td>Perez</td>
                    </tr>
                    <tr>
                        <td>5</td>
                        <td>
                            <em className = "symm"> </em>
                            perez
                            </td>
                        <td>
                            <em className = "sym"> </em>
                            verstappen
                        </td>
                        <td>
                            <em className = "sym"> </em>
                            verstappen
                        </td>
                        <td>TBD!</td>
                    </tr>
                </table>

                <div className = "legend">
                    <h2>&#x2714; — earned money</h2>
                    <h2>&#x2718; — lost money</h2>
                </div>

            </div>

            <div className = "divide-white-short center"></div>


            <div className = "preds-h3 betting-log center piece">
                <h2>Betting Log Visualization</h2>
                <h3 className = "preds-h3">We are also keeping tracking of our hypothetical bets to see how much money we would have
                earned (or lost) in theory using our models' predictions by betting strategy. We plot the
                percent change for each bet, and we visualize the earnings and losses per betting strategy of
                interest to compare them.
                </h3>
                <div className = "line-graph center">
                    <BetsMoney />
                </div>
                <div className = "line-graph">
                </div>

            </div>

            <Footer/>

            </div>
        </>



    )
}

export default Predictions;
