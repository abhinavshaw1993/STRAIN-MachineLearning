x -

SELECT activity_time ,student_id ,activity_inference FROM activity_details

SELECT dinning_time ,student_id	,venue_id ,meal_type FROM dinning_details

select call_time, student_id, call_duration_min, call_type from call_log_details

select timestamp, student_id, 1 from sms_details

select audio_activity_time, student_id, audio_activity_inference from audio_details

select conv_start_timestamp, student_id, conv_duration_min  from conversation_details

select dark_start_timestamp, student_id, dark_duration_min from dark_details

select start_timestamp, student_id, phonecharge_duration_min from phonecharge_details

select start_timestamp, student_id, phonelock_duration_min from phonelock_details

select wifi_timestamp as time, student_id, latitude, longitude from gps_details

select cast(callender_date as datetime) + cast(callender_time as datetime), student_id, account_label from calender_details


y -

select response_time, student_id, stress_level from stress_details