# PyMARL-Multi SMAC Map Edit

Edit map and create RL units as follows:

## Copy *_multi.SC2Map and edit the copied map file

## Edit Unit

1. Open editor, data editor, unit tab
2. Right click and click **add new unit**
3. **Name** the new unit, click suggest right below it
4. Leave the "parent:" row alone. That determines what we're making. We want to make a unit
5. **Select the unit** you want to copy (bottom of the new opened window, "copy from" row) e.g. zealot if you're copying zealot
6. Set the **"Object family:," "Race:,"** and **"Object Type:"** as desired. THESE DO NOTHING but make it easier for you to find your new unit once it's made. e.g. you probably want a new zerg unit to be in the zerg section when you go to place it on your map or something.
7. Press okay
8. Go back to the Unit tab, find the new unit and modify the following fields:
- (Basic) Stats: Supplies - **0**
- Combat: Default Acquire Level - **Passive**
- Behaviour: Response - **No Response**

## Edit Actor

1. Click the plus sign on the data editor tabs, go to edit actor data, actors
2. Click the new actors tab
3. Right click and click add new actor
4. Name it and click suggest like before
5. Change the **"Actor Type:" row to unit**
6. Select what you want to copy from (bottom of the new opened window again) e.g. zealot if you're coping a zealot
7. Press okay
8. Click on your new actor
9. At the **bottom right** of the window where it says **"Token"** and then **"Unit Name,"** change the unit name to the name of your unit e.g. Zealot RL

## Edit smac_maps.py and add "_multi" item just like other map items



