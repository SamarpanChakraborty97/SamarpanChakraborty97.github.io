-- Use the database required for analyzing the student engagement across two quarters
USE data_scientist_project;

###########################################################
#### LOOK AT THE DIFFERENT TABLES INSIDE THE DATABASE #####
###########################################################
-- STUDENT CERTIFICATES TABLE
SELECT 
    *
FROM
    student_certificates
LIMIT 10;

## STUDENT_INFO TABLE
SELECT 
    *
FROM
    student_info
LIMIT 10;

## STUDENT_PURCHASES TABLE
SELECT 
    *
FROM
    student_purchases
LIMIT 10;

## STUDENT_VIDEO_WATCHED TABLE
SELECT 
    *
FROM
    student_video_watched;
    
########################################################
######################################################## 

SELECT 
    student_id, SUM(seconds_watched) AS total_seconds
FROM
    student_video_watched
WHERE
    date_watched BETWEEN '2021-04-01' AND '2021-06-30'
GROUP BY student_id;

################################################################################
##### CREATE A NEW VIEW CONTAINING INFORMATION ABOUT STUDENT SUBSCRIPTION ######
################################################################################
CREATE VIEW subscription_info AS
    SELECT 
        purchase_id,
        student_id,
        plan_id,
        date_purchased AS date_start,
        CASE -- Start of CASE statement to handle different subscription plan durations
            WHEN
                date_refunded IS NULL
            THEN
                (SELECT 
                        CASE
                                WHEN
                                    plan_id = 0
                                THEN
                                    DATE_ADD(date_purchased,
                                        INTERVAL 1 MONTH) -- If 'plan_id' is 0 (monthly subscription), then add one month to 'date_purchased' to calculate 'date_end'
                                WHEN
                                    plan_id = 1
                                THEN
                                    DATE_ADD(date_purchased,
                                        INTERVAL 3 MONTH) -- If 'plan_id' is 1 (quarterly subscription), then add three months to 'date_purchased' to calculate 'date_end'
                                WHEN
                                    plan_id = 2
                                THEN
                                    DATE_ADD(date_purchased,
                                        INTERVAL 12 MONTH) -- If 'plan_id' is 2 (annual subscription), then add twelve months to 'date_purchased' to calculate 'date_end'
                                ELSE NULL -- No end date
                            END
                    )
            ELSE date_refunded
        END AS date_end
    FROM
        student_purchases;
 ###############################################################################
 ###############################################################################
 
################################################################################
####### CREATE A NEW VIEW CONTAINING INFORMATION ABOUT STUDENT PURCHASE ########
################################################################################
drop view if exists purchase_info;
CREATE VIEW purchase_info AS
    SELECT 
        student_id,
        CASE
            WHEN date_end < '2021-04-01'
            THEN
                0
            WHEN date_start > '2021-06-30' THEN 0
            ELSE 1
        END AS paid_q2_2021,
        CASE
            WHEN date_end < '2022-04-01'
            THEN
                0
            WHEN date_start > '2022-06-30' THEN 0
            ELSE 1
        END AS paid_q2_2022
    FROM
        subscription_info;
 ###############################################################################
 ###############################################################################  
 
 #########################################################
 #### CALCULATING THE MINUTES WATCHED IN 2021 or 2022 ####
 #########################################################
 SELECT 
    student_id,
    -- total seconds watched is converted to minutes and rounded to 2 decimal places
    ROUND(SUM(seconds_watched) / 60, 2) AS minutes_watched
FROM
    student_video_watched
WHERE
	-- !!! 2021 or 2022 dpending on the year referenced
    YEAR(date_watched) = 2021
-- Grouping results by each student to get aggregate minutes watched over the year
GROUP BY student_id;
###########################################################
###########################################################

##########################################################################
######## SELECTING THE STUDENTS PAYING IN Q2 OF A PARTICULAR YEAR ########
##########################################################################
SELECT 
  min_watched.student_id, 
  min_watched.minutes_watched, 
  IF(
    income.date_start IS NULL, -- to check if the student has a start date in purchases_info
    0, 
    MAX(income.paid_q2_2022) -- !!! to be changed to 2021 or 2022 depending on the year considered 
  ) AS paid_in_q2 
FROM 
  (-- Subquery to get total minutes watched by each student for a specific year
    SELECT 
      student_id, 
      
      -- Convert total seconds watched to minutes and round to 2 decimal places
      ROUND(
        SUM(seconds_watched) / 60, 
        2
      ) AS minutes_watched 
    FROM 
      student_video_watched 
    WHERE 
      YEAR(date_watched) = 2022 -- Ensure consistency with paid_q2 year 
    GROUP BY 
      student_id
  ) min_watched 
  LEFT JOIN purchases_info income ON min_watched.student_id = income.student_id 
GROUP BY 
  student_id
HAVING paid_in_q2 = 0; -- !!! to be changed to 0 or 1 based on desired filter 
################################################################################ 
################################################################################

############################################
#####  NUMBER OF CERTIFICATES ISSUED  ######
############################################
SELECT 
    cert.student_id,
    cert.num_certificates,
    IFNULL(vid.minutes_watched, 0) as minutes_watched
FROM
    (SELECT 
        student_id, COUNT(student_id) AS num_certificates
    FROM
        student_certificates
    GROUP BY student_id) cert
        LEFT JOIN
    (SELECT 
        student_id,
            ROUND(SUM(seconds_watched / 60), 2) AS minutes_watched
    FROM
        student_video_watched
    GROUP BY student_id) vid ON cert.student_id = vid.student_id
ORDER BY cert.student_id;
##############################################
##############################################

###################################################
##### CREATING THE REQIUIRED CSV FILES LATER ######
###################################################

--  Calculating the number of students who watched a lecture in Q2 2021
SELECT 
    COUNT(DISTINCT student_id)
FROM
    student_video_watched
WHERE
    YEAR(date_watched) = 2021;
    

-- Calculating the number of students who watched a lecture in Q2 2022
SELECT 
    COUNT(DISTINCT student_id)
FROM
    student_video_watched
WHERE
    YEAR(date_watched) = 2022;
    

-- Calculating the number of students who watched a lecture in Q2 2021 and Q2 2022
SELECT 
    COUNT(DISTINCT student_id)
FROM
    (
    -- Subquery to get unique students who watched lectures in 2021
    SELECT DISTINCT
        student_id
    FROM
        student_video_watched
    WHERE
        YEAR(date_watched) = 2021) a -- Alias for the subquery results for 2021
	-- Join with another subquery to get students who also watched videos in 2022
	JOIN 
    (
    -- Subquery to get unique students who watched videos in 2022
    SELECT DISTINCT
        student_id
    FROM
        student_video_watched
    WHERE
        YEAR(date_watched) = 2022) b -- Alias for the subquery results for 2022
	-- Specify the common column (student_id) for joining the results of the two subqueries
	USING(student_id);
    
            
-- Calculating the total number of students who watched a lecture
SELECT 
    COUNT(DISTINCT student_id)
FROM
    student_video_watched;

##################################################################################
##################################################################################