# CutoffPredictor
Tool for water utilities to monitor and predict customers' risk of service interruption

## Inputs
1. Database containing the following tables (with the minimum necessary set of columns):
 - meter:
   - meter_id = meter id code
   - location_id = meter location id code (foreign key)
   - cust_type = customer type (string)
   - cust_type_code = customer type code (int)
   - meter_size = meter size (float)
   - (note that we don't care about meter installation dates for this purpose; we assume that the meter size and customer type are the same for all meters installed at a particular address)
 - meter_location:
   - location_id = meter location id code
   - latitude = latitude (float)
   - longitude = longitude (float)
   - meter_address = meter_address (string)
   - municipality = municipality code (separate code for each city) (int)
 - occupant:
   - location_id = meter location id code (foreign key)
   - cis_occupant_id = suffix to add to location_id to get unique occupant id
   - move_in_date = date moved in (date)
   - move_out_date = date moved out (date)
 - volume:
   - meter_id = meter id code
   - meter_read_at = date on which the meter was read (date)
   - volume_kgals = volume consumed in kGal over the interval between meter_read_at of this record and meter_read_at of the previous record (or since the start of service) (float)
 - charge:
   - location_id = meter location id code (foreign key)
   - cis_occupant_id = suffix to add to location_id to get unique occupant id
   - billing_period_end = date on which the bill was computed (date)
   - total_charge = total charge (in $) for usage, including baseline and usage-based charges (float)
   - late_charge = extra charge (in $) for late payment, if any (float)
 - cutoffs:
   - location_id = meter location id code (foreign key)
   - cis_occupant_id = suffix to add to location_id to get unique occupant id
   - cutoff_at = date of cutoff (date)

2. Configuration file:
 - contains information about:
   - paths to locations of code and data
   - runtime options (which stage of processing to run)
   - database access parameters
   - mapping access parameters
   - model types and feature engineering parameters
   - date on which to make the prediction



